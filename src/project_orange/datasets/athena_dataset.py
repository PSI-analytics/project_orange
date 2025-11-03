"""``AthenaQueryDataset`` to load and save data from Athena."""

from __future__ import annotations

from typing import Any, NoReturn

import pandas as pd
from kedro.io import AbstractDataset, DatasetError
import boto3


class AthenaQueryDataset(AbstractDataset[None, pd.DataFrame]):
    """Loads data from a provided SQL query from an Athena dataset hosted on AWS"""

    def __init__(
        self,
        sql: str = None,
        s3_staging_dir: str = None,
        database: str = None,
        aws_region: str = None,
    ) -> None:
        """
        sql: SQL query to extract data from table.
        s3_staging_dir: Where to direct the Athena output - an S3 path to the output folder.
        database: database table
        aws_region: AWS Region
        """
        self.sql_query = sql
        self.s3_staging_dir = s3_staging_dir
        self.database = database
        self.region_name = aws_region

    def _run_query(self) -> str:
        """
        Run the SQL query on AWS Athena, and return the query execution ID, which can be used to track the query and
        extract the data once complete.
        :return: Query execution ID
        """
        response = self.athena_client.start_query_execution(
            QueryString=self.sql_query,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.s3_staging_dir},
        )
        return response["QueryExecutionId"]

    def _get_query_results(self, query_execution_id: str) -> pd.DataFrame:
        """
        The method that is called whilst we wait for the Athena query to run, and once it has completed returns the
        output dataframe.
        :param query_execution_id: The ID of the Athena query just ran, which will enable us to track the status of the
        query, and give us the filepath of the output once complete.
        :return: Dataframe of the Athena output
        """
        # We can track the status of our query, which will return 'QUEUED', 'RUNNING', 'SUCCEEDED', or an error
        response = self.athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        while response["QueryExecution"]["Status"]["State"] in ("QUEUED", "RUNNING"):
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )

        # Once complete, we can pull the outputted data from the Athena output file location
        if response["QueryExecution"]["Status"]["State"] == "SUCCEEDED":
            s3_path = response["QueryExecution"]["ResultConfiguration"][
                "OutputLocation"
            ]
            return pd.read_csv(f"s3://{s3_path}")

        else:
            if "AthenaError" in response["QueryExecution"]["Status"].keys():
                error_message = response["QueryExecution"]["Status"]["AthenaError"][
                    "ErrorMessage"
                ]
            else:
                error_message = response["QueryExecution"]["Status"]
            raise Exception(
                "Query failed to run with state: {}\n{}".format(
                    response["QueryExecution"]["Status"]["State"], error_message
                )
            )

    def _load(self) -> pd.DataFrame:
        """
        The method that is called when the dataset class is used.
        :return: Dataframe of data extracted from the Athena table using the SQL query.
        """
        # Create an Athena session with boto3
        session = boto3.Session(region_name=self.region_name)
        self.athena_client = session.client("athena")

        # Run the query using self.sql_query
        query_execution_id = self._run_query()
        output_df = self._get_query_results(query_execution_id)

        return output_df

    def _save(self, data: None) -> NoReturn:
        raise DataSetError("'save' is not supported on AthenaQueryDataset")

    def _describe(self) -> dict[str, Any]:
        return dict()
