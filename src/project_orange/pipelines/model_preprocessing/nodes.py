"""
This is a boilerplate pipeline 'model_preprocessing'
generated using Kedro 1.0.0
"""

import logging
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import seaborn as sns

from ..data_schemas.data_schemas import (
    attendance_model_schema,
    viewership_model_schema,
)

logger = logging.getLogger(__name__)


def _clean_attendance_data(
    attendance_df: pd.DataFrame,
):
    """
    Cleans and standardizes a raw attendance DataFrame for football matches.

    This function performs the following steps:
    1. Converts the "Date" column to a datetime object.
    2. Selects a subset of relevant columns.
    3. Renames columns to standardized, snake_case names.
    4. Fills missing kick-off times with the mode of that column.
    5. Sorts the DataFrame by match date and kick-off time.
    6. Resets the index.

    Args:
        attendance_df (pd.DataFrame): Raw attendance DataFrame containing at least the
            following columns: "Season", "Date", "Kick-Off Time", "Home", "Away",
            "Stadium", "Capacity", "Attendance", "Utilisation".

    Returns:
        pd.DataFrame: Cleaned attendance DataFrame with standardized column names and
        sorted by match date and kick-off time.
    """
    # Convert 'Date' column to datetime
    attendance_df["match_date"] = pd.to_datetime(attendance_df["Date"])

    # clean names in the data
    attendance_df["Home"] = attendance_df["Home"].str.strip()
    attendance_df["Away"] = attendance_df["Away"].str.strip()
    attendance_df["Stadium"] = attendance_df["Stadium"].str.strip()

    # clean typos in the data
    # Mapping dictionary
    mapping = {
        "al kholood": "Al Kholood",
        "Al kholood": "Al Kholood",
        "Al khaleej": "Al Khaleej",
        "al fayha": "Al Fayha",
        "Al Weha": "Al Wehda",
        "Al Orobah": "Al Urooba",
    }

    # Apply the mapping
    attendance_df["Home"] = attendance_df["Home"].replace(mapping)
    attendance_df["Away"] = attendance_df["Away"].replace(mapping)

    # Select relevant columns and create a copy to avoid SettingWithCopyWarning
    clean_attendance_df = attendance_df[
        [
            "Season",
            "match_date",
            "Kick-Off Time",
            "Home",
            "Away",
            "Stadium",
            "Capacity",
            "Attendance",
            "Utilisation",
        ]
    ].copy()

    # Rename columns to standardized names
    clean_attendance_df.rename(
        columns={
            "Season": "season",
            "Kick-Off Time": "match_kick_off_time",
            "Home": "home_team",
            "Away": "away_team",
            "Stadium": "venue_name",
            "Capacity": "venue_capacity",
            "Attendance": "attendance",
            "Utilisation": "venue_percentage_capacity",
        },
        inplace=True,
    )

    # Fill missing kick-off times with the mode of the column
    if clean_attendance_df["match_kick_off_time"].isnull().any():
        clean_attendance_df["match_kick_off_time"].fillna(
            clean_attendance_df["match_kick_off_time"].mode()[0], inplace=True
        )

    # Sort by match date and kick-off time
    clean_attendance_df = clean_attendance_df.sort_values(
        by=["match_date", "match_kick_off_time"],
    ).reset_index(drop=True)

    # manual update to one of the games which had the wrong away team
    # https://www.spl.com.sa/en/match/84140
    clean_attendance_df.loc[
        (clean_attendance_df["season"] == "2024/2025")
        & (clean_attendance_df["home_team"] == "Al Ettifaq")
        & (clean_attendance_df["away_team"] == "Al Okhdoud")
        & (clean_attendance_df["match_date"] == "2024-08-28"),
        "away_team",
    ] = "Al Kholood"

    return clean_attendance_df


def _validate_with_schema(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """
    Validates a DataFrame using a given pandera schema and reorders the columns
    to match the schema definition.

    Args:
        df (pd.DataFrame): Input DataFrame to be validated.
        schema (pa.DataFrameSchema): Pandera schema to validate against.

    Returns:
        pd.DataFrame: Validated DataFrame with columns ordered as per schema.

    Raises:
        pa.errors.SchemaError: If the DataFrame does not conform to the schema.
    """
    try:
        validated_df = schema.validate(df)
    except pa.errors.SchemaError as e:
        raise e

    # Reorder the columns based on the schema
    schema_columns = list(schema.to_schema().columns)
    return validated_df[schema_columns]


def _aggregate_player_and_team_data(
    player_elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cleans team names and aggregates player Elo data at the match-team level.

    This function performs the following steps:
    1. Strips leading/trailing spaces from `team_name` and `opposition_name`.
    2. Groups the data by `match_id`, `team_id`, and `opposition_id`.
    3. Aggregates key statistics using either first occurrence or sum as appropriate.
    4. Converts the `date` column to a new `match_date` column as datetime.
    5. Applies a name mapping dictionary to standardize team names.

    Args:
        player_elo_df (pd.DataFrame): Input DataFrame containing player-level Elo data.
            Must include at least the following columns:
            - "match_id", "team_id", "opposition_id"
            - "date", "competition_id", "competition_name"
            - "wyscout_season_id", "season_name"
            - "team_name", "opposition_name"
            - "team_pre_match_rating", "opposition_pre_match_rating"
            - "average_team_rating_career", "minutes_played"
        name_mapper (dict): Mapping dictionary used to standardize team names.
            Example: {"al kholood": "Al Kholood", "al khaleej": "Al Khaleej"}

    Returns:
        pd.DataFrame: Aggregated DataFrame at the match-team level with standardized team names
        and an added `match_date` column.
    """
    # Clean team name whitespace
    player_elo_df["team_name"] = player_elo_df["team_name"].str.strip()
    player_elo_df["opposition_name"] = player_elo_df["opposition_name"].str.strip()

    player_elo_df["weighted_minutes_played"] = player_elo_df["minutes_played"] / 90

    # Update the column: if > 1, set to 1; else leave as is
    player_elo_df.loc[
        player_elo_df["weighted_minutes_played"] > 1, "weighted_minutes_played"
    ] = 1

    player_elo_df["weighted_squad_rating"] = (
        player_elo_df["weighted_minutes_played"]
        * player_elo_df["average_team_rating_career"]
    )

    # Aggregate player data to team-match level
    player_and_team_df = (
        player_elo_df.groupby(["match_id", "team_id", "opposition_id"])
        .agg(
            {
                "date": "first",
                "competition_id": "first",
                "competition_name": "first",
                "wyscout_season_id": "first",
                "season_name": "first",
                "match_id": "first",
                "team_id": "first",
                "team_name": "first",
                "opposition_id": "first",
                "opposition_name": "first",
                "team_pre_match_rating": "first",
                "opposition_pre_match_rating": "first",
                "average_team_rating_career": "sum",
                "minutes_played": "sum",
                "weighted_minutes_played": "sum",
                "weighted_squad_rating": "mean",
            }
        )
        .sort_values(by="date")
        .reset_index(drop=True)
    )

    # Add match_date column
    player_and_team_df["match_date"] = pd.to_datetime(player_and_team_df["date"])

    latest_team_ratings_df = player_and_team_df.loc[
        player_and_team_df.groupby("team_name")["date"].idxmax()
    ]

    latest_team_ratings_df = (
        latest_team_ratings_df[
            [
                "date",
                "team_name",
                "team_pre_match_rating",
                "weighted_squad_rating",
            ]
        ]
        .sort_values(by=["date", "team_name"])
        .reset_index(drop=True)
    )

    latest_team_ratings_df = latest_team_ratings_df.rename(
        columns={
            "team_pre_match_rating": "prospect_team_rating",
            "weighted_squad_rating": "prospect_squad_rating",
            "date": "latest_date_data",
        }
    )

    return player_and_team_df, latest_team_ratings_df


def _add_in_time_slots(
    attendance_model_df: pd.DataFrame,
    kick_off_time="match_kick_off_time",
) -> pd.DataFrame:
    """adds in times slots to the attendance model dataframe

    Args:
        attendance_model_df: dataframe for attendance model

    Returns:
        attendance_model_df_with_timeslots: returns updated attendance model with time slots added in

    """

    attendance_model_df_with_timeslots = attendance_model_df

    time_slot_choices = [
        attendance_model_df_with_timeslots["day_of_week"] == "Monday",
        attendance_model_df_with_timeslots["day_of_week"] == "Tuesday",
        (attendance_model_df_with_timeslots["day_of_week"] == "Wednesday"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Thursday")
        & (attendance_model_df_with_timeslots[kick_off_time].astype(str) == "18:00:00"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Thursday")
        & (attendance_model_df_with_timeslots[kick_off_time].astype(str) == "21:00:00"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Thursday"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Friday")
        & (attendance_model_df_with_timeslots[kick_off_time].astype(str) == "18:00:00"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Friday")
        & (attendance_model_df_with_timeslots[kick_off_time].astype(str) == "21:00:00"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Friday"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Saturday")
        & (attendance_model_df_with_timeslots[kick_off_time].astype(str) == "18:00:00"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Saturday")
        & (attendance_model_df_with_timeslots[kick_off_time].astype(str) == "21:00:00"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Saturday"),
        (attendance_model_df_with_timeslots["day_of_week"] == "Sunday"),
    ]

    # these have been chosen with a roughly even split across timeslots in mind
    time_slot_outcomes = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday 18:00",
        "Thursday 21:00",
        "Thursday Other",
        "Friday 18:00",
        "Friday 21:00",
        "Friday Other",
        "Saturday 18:00",
        "Saturday 21:00",
        "Saturday Other",
        "Sunday",
    ]

    attendance_model_df_with_timeslots["time_slot"] = np.select(
        time_slot_choices,
        time_slot_outcomes,
        default="other",
    )

    # useful variable to evaluate when checking distribution of timeslots
    time_slot_counts = attendance_model_df_with_timeslots["time_slot"].value_counts()

    return attendance_model_df_with_timeslots


def _extract_date_features(
    df: pd.DataFrame,
    date_column="date",
    month_column="month",
    day_of_week_column="day_of_week",
    week_of_month_column="week_of_month",
):
    """Extract month, day of week, and week of month from a datetime column.

    This function takes a DataFrame with a date column and creates three new columns:
    one containing the month name, one containing the day of week name, and one
    containing the week of month number (1-5).

    Args:
        df (pd.DataFrame): The input DataFrame containing the date column.
        date_column (str, optional): Name of the column containing dates.
            Defaults to 'date'.
        month_column (str, optional): Name of the new column for months.
            Defaults to 'month'.
        day_of_week_column (str, optional): Name of the new column for day of week.
            Defaults to 'day_of_week'.
        week_of_month_column (str, optional): Name of the new column for week of month.
            Defaults to 'week_of_month'.

    Returns:
        pd.DataFrame: The DataFrame with three additional columns containing month names,
            day of week names, and week of month numbers.

    Examples:
        >>> df = pd.DataFrame({"date": ["2023-01-15", "2023-02-20"]})
        >>> df["date"] = pd.to_datetime(df["date"])
        >>> result = _extract_date_features(df)
        >>> result[["month", "day_of_week", "week_of_month"]].values.tolist()
        [['January', 'Sunday', 3], ['February', 'Monday', 3]]
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_column])
    df[month_column] = dt.dt.month_name()
    df[day_of_week_column] = dt.dt.day_name()
    df[week_of_month_column] = np.ceil(dt.dt.day / 7).astype(int)
    return df


def _add_in_derbies(
    df: pd.DataFrame,
    derby_pairs: dict,
    home_column="home_team",
    away_column="away_team",
    derby_column="derby",
):
    """Add a binary column indicating whether a match is a derby.

    This function identifies derby matches based on specified team pairs and adds
    a binary column to the DataFrame. A match is marked as a derby (1) if the
    home and away teams match any derby pair in either order, otherwise it's
    marked as 0.

    Args:
        df (pd.DataFrame): The input DataFrame containing home and away team columns.
        derby_pairs (dict): Dictionary where keys and values represent derby pairs.
            Each key-value pair defines teams that form a derby. Order is irrelevant
            as both (team_1, team_2) and (team_2, team_1) are considered derbies.
        home_column (str, optional): Name of the home team column. Defaults to 'home_team'.
        away_column (str, optional): Name of the away team column. Defaults to 'away_team'.
        derby_column (str, optional): Name of the new derby indicator column.
            Defaults to 'derbies'.

    Returns:
        pd.DataFrame: The DataFrame with an additional binary column indicating derbies.

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "home_team": ["Arsenal", "Arsenal", "Chelsea"],
        ...         "away_team": ["Tottenham", "Chelsea", "Arsenal"],
        ...     }
        ... )
        >>> derby_pairs = {"Arsenal": "Tottenham", "Liverpool": "Everton"}
        >>> result = add_derbies_column(df, derby_pairs)
        >>> result["derbies"].tolist()
        [1, 0, 0]

        >>> # Multiple derby pairs example
        >>> derby_pairs = {"Arsenal": "Tottenham", "Arsenal": "Chelsea"}
        >>> result = add_derbies_column(df, derby_pairs)
        >>> result["derbies"].tolist()
        [1, 1, 1]
    """
    df = df.copy()

    # Create a set of frozensets for efficient lookup (order-independent)
    derby_set = {frozenset([team1, team2]) for team1, team2 in derby_pairs.items()}

    # Vectorized check: create frozenset for each row and check membership
    df[derby_column] = df.apply(
        lambda row: int(frozenset([row[home_column], row[away_column]]) in derby_set),
        axis=1,
    )

    return df


def _get_latest_leverage_simulations(
    df: pd.DataFrame,
    leverage_variable: Union[str, List[str]],
    match_id: Optional[int] = None,
) -> pd.DataFrame:
    """Retrieve the latest simulation data for given leverage variable(s).

    Filters the input dataframe to return only the most recent simulate_from_date
    entries for each match_id and leverage_variable combination. Optionally
    filters to a specific match_id.

    Args:
        df: Input dataframe containing match simulation data. Must include columns:
            'match_id', 'leverage_variable', 'simulate_from_date', and
            'match_jeopardy_wrt_finish'.
        leverage_variable: The leverage variable(s) to filter by. Can be a single
            string (e.g., 'top_1') or a list of strings (e.g., ['top_1', 'top_5']).
        match_id: Optional specific match_id to filter. If None, returns latest
            simulations for all match_ids with the given leverage_variable(s).

    Returns:
        A filtered dataframe containing only the latest simulate_from_date records
        for each match_id and leverage_variable combination. Returns an empty
        dataframe if no matches are found.

    Raises:
        KeyError: If required columns are missing from the input dataframe.
        ValueError: If leverage_variable is empty, None, or an empty list.

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "match_id": [5758599, 5758599, 5758597, 5758597, 5758599],
        ...         "leverage_variable": ["top_1", "top_1", "top_1", "top_5", "top_5"],
        ...         "simulate_from_date": [
        ...             "2025/08/28",
        ...             "2025/08/27",
        ...             "2025/08/28",
        ...             "2025/08/28",
        ...             "2025/08/27",
        ...         ],
        ...         "match_jeopardy_wrt_finish": [
        ...             0.044854,
        ...             0.035123,
        ...             0.092689,
        ...             0.015432,
        ...             0.012345,
        ...         ],
        ...     }
        ... )
        >>> result = get_latest_leverage_simulations(df, "top_1")
        >>> len(result)
        2

        >>> result = get_latest_leverage_simulations(df, ["top_1", "top_5"])
        >>> len(result)
        3

        >>> result = get_latest_leverage_simulations(df, ["top_1"], match_id=5758599)
        >>> len(result)
        1
    """
    # Validate inputs
    if leverage_variable is None:
        raise ValueError("leverage_variable cannot be None")

    # Convert single string to list for uniform processing
    if isinstance(leverage_variable, str):
        if leverage_variable == "":
            raise ValueError("leverage_variable must be a non-empty string")
        leverage_variables = [leverage_variable]
    elif isinstance(leverage_variable, list):
        if len(leverage_variable) == 0:
            raise ValueError("leverage_variable list cannot be empty")
        if any(not isinstance(lv, str) or lv == "" for lv in leverage_variable):
            raise ValueError("All leverage_variable values must be non-empty strings")
        leverage_variables = leverage_variable
    else:
        raise ValueError("leverage_variable must be a string or list of strings")

    required_cols = [
        "match_id",
        "leverage_variable",
        "simulate_from_date",
        "match_jeopardy_wrt_finish",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Create a copy to avoid modifying original dataframe
    df_filtered = df.copy()

    # Filter by leverage_variable(s)
    df_filtered = df_filtered[df_filtered["leverage_variable"].isin(leverage_variables)]

    # Filter by match_id if provided
    if match_id is not None:
        df_filtered = df_filtered[df_filtered["match_id"] == match_id]

    # Return empty dataframe if no matches
    if df_filtered.empty:
        return df_filtered

    # Convert simulate_from_date to datetime for proper comparison
    df_filtered["simulate_from_date"] = pd.to_datetime(
        df_filtered["simulate_from_date"], format="%Y/%m/%d"
    )

    # Get the latest simulate_from_date for each match_id and leverage_variable combination
    idx = df_filtered.groupby(["match_id", "leverage_variable"])[
        "simulate_from_date"
    ].idxmax()
    result = df_filtered.loc[idx].reset_index(drop=True)

    # Convert date back to original format
    result["simulate_from_date"] = result["simulate_from_date"].dt.strftime("%Y/%m/%d")

    # Select columns needed for pivoting and final output
    base_cols = [
        "competition_id",
        "competition_name",
        "season",
        "match_id",
        "home_team_name",
        "away_team_name",
        "home_team_win_prob",
        "away_team_win_prob",
    ]
    pivot_cols = base_cols + ["leverage_variable", "match_jeopardy_wrt_finish"]
    result = result[pivot_cols]

    # Pivot the dataframe to transform leverage_variable rows into columns
    result_pivoted = result.pivot_table(
        index=base_cols,
        columns="leverage_variable",
        values="match_jeopardy_wrt_finish",
        aggfunc="first",
    ).reset_index()

    # Rename the pivoted columns to match_jeopardy_{leverage_variable}
    result_pivoted.columns = [
        f"match_jeopardy_{col}" if col not in base_cols else col
        for col in result_pivoted.columns
    ]

    return result_pivoted


def _merge_team_ratings(
    df: pd.DataFrame,
    elo_df: pd.DataFrame,
    home_col: str = "home_team_name",
    away_col: str = "away_team_name",
) -> pd.DataFrame:
    """Merges home and away team ELO ratings into a match-level DataFrame.

    This function adds the team's pre-match rating and average squad rating
    for both the home and away teams in each match.

    Args:
        df (pd.DataFrame): Match-level DataFrame containing columns for match ID,
            home team, and away team.
        elo_df (pd.DataFrame): DataFrame containing team ELO ratings with columns:
            'match_id', 'team_name', 'team_pre_match_rating', 'average_team_rating_career'.
        home_col (str, optional): Name of the column containing the home team. Defaults to "home_team_name".
        away_col (str, optional): Name of the column containing the away team. Defaults to "away_team_name".

    Returns:
        pd.DataFrame: DataFrame with additional columns:
            - team_pre_match_rating_x (home team rating)
            - team_pre_match_rating_y (away team rating)
            - average_team_rating_career_x (home squad rating)
            - average_team_rating_career_y (away squad rating)
    """
    df = df.merge(
        elo_df[
            [
                "match_id",
                "match_date",
                "team_name",
                "team_pre_match_rating",
                "average_team_rating_career",
            ]
        ],
        left_on=["match_id", home_col],
        right_on=["match_id", "team_name"],
        how="left",
    )
    df = df.merge(
        elo_df[
            [
                "match_id",
                "team_name",
                "team_pre_match_rating",
                "average_team_rating_career",
            ]
        ],
        left_on=["match_id", away_col],
        right_on=["match_id", "team_name"],
        how="left",
    )
    return df


def _create_single_prospect_df(
    jeopardy_df: pd.DataFrame,
    team_and_player_elo_df: pd.DataFrame,
    match_id_mapper: dict,
) -> pd.DataFrame:
    """Creates a prospect-level performance DataFrame by merging match data with team and player ELO ratings.

    This function merges home and away team ELO ratings into the given match data (`jeopardy_df`),
    handles missing or inconsistent `match_id` values using a mapping dictionary, and calculates
    combined and differential team and squad ratings for each match.

    Args:
        jeopardy_df (pd.DataFrame): DataFrame containing match-level data, including 'match_id',
            'home_team_name', and 'away_team_name'.
        team_and_player_elo_df (pd.DataFrame): DataFrame containing team and player ELO ratings with columns
            'match_id', 'team_name', 'team_pre_match_rating', and 'average_team_rating_career'.
        match_id_mapper (dict): Dictionary mapping incorrect `match_id` values to correct ones.

    Returns:
        pd.DataFrame: Enriched DataFrame containing original match data along with:
            - prospect_home_team_rating
            - prospect_away_team_rating
            - prospect_home_squad_rating
            - prospect_away_squad_rating
            - prospect_combined_team_rating
            - prospect_team_rating_abs_difference
            - prospect_combined_squad_rating
    """

    # Initial merge
    prospect_df = _merge_team_ratings(jeopardy_df, team_and_player_elo_df)

    # Handle missing ratings due to bad match_ids
    missing_mask = prospect_df["team_pre_match_rating_x"].isna()
    if missing_mask.any():
        corrected_elo_df = team_and_player_elo_df.copy()
        corrected_elo_df["match_id"] = corrected_elo_df["match_id"].replace(
            match_id_mapper
        )

        # Re-merge only the problematic rows
        na_df = (
            prospect_df.loc[missing_mask]
            .drop(
                [
                    "team_pre_match_rating_x",
                    "team_pre_match_rating_y",
                    "team_name_x",
                    "team_name_y",
                    "match_date",
                    "average_team_rating_career_x",
                    "average_team_rating_career_y",
                ],
                axis=1,
            )
            .reset_index()
        )

        na_df = _merge_team_ratings(na_df, corrected_elo_df)
        na_df = na_df.set_index("index")
        prospect_df.loc[na_df.index, na_df.columns] = na_df

    # Rename columns for clarity
    prospect_df = prospect_df.rename(
        columns={
            "team_pre_match_rating_x": "prospect_home_team_rating",
            "team_pre_match_rating_y": "prospect_away_team_rating",
            "average_team_rating_career_x": "prospect_home_squad_rating",
            "average_team_rating_career_y": "prospect_away_squad_rating",
        }
    ).drop(columns=["team_name_x", "team_name_y"])

    # Compute combined and differential ratings
    prospect_df["prospect_combined_team_rating"] = (
        prospect_df["prospect_home_team_rating"]
        + prospect_df["prospect_away_team_rating"]
    )
    prospect_df["prospect_team_rating_abs_difference"] = abs(
        prospect_df["prospect_home_team_rating"]
        - prospect_df["prospect_away_team_rating"]
    )
    prospect_df["prospect_combined_squad_rating"] = (
        prospect_df["prospect_home_squad_rating"]
        + prospect_df["prospect_away_squad_rating"]
    )

    return prospect_df


def _clean_viewership_df(
    df: pd.DataFrame, match_col: str, viewership_col: str, name_mapper: dict = None
) -> pd.DataFrame:
    """Cleans a viewership DataFrame by standardizing columns, removing unwanted rows,
    splitting matches into home and away teams, and cleaning team names.

    Args:
        df (pd.DataFrame): Raw viewership DataFrame to clean.
        match_col (str): Name of the column containing the match information.
        viewership_col (str): Name of the column containing the viewership numbers.
        name_mapper (dict, optional): Dictionary mapping team names to standardized names. Defaults to None.

    Returns:
        pd.DataFrame: Cleaned viewership DataFrame with columns ['match', 'viewership', 'home_team', 'away_team'].
    """
    df = df.dropna(axis=0, how="all").copy()

    # Rename columns
    df = df.rename(columns={match_col: "match", viewership_col: "viewership"})

    # Drop rows with missing matches or unwanted labels
    df = df[df["match"].notna()]
    df = df[
        ~df["match"].isin(["Match", "Magazine Coverage", "Market - KSA"])
    ].reset_index(drop=True)

    # Convert viewership to int
    df["viewership"] = df["viewership"].astype(int)

    # Split match into home and away teams
    df[["home_team", "away_team"]] = df["match"].str.split(" vs ", expand=True)

    # Clean team names
    suffixes = r"\b(SFC|FC|SC|Club)\b"
    for col in ["home_team", "away_team"]:
        df[col] = (
            df[col]
            .str.strip()
            .str.replace(suffixes, "", regex=True, case=False)
            .str.strip()
        )
        if name_mapper:
            df[col] = df[col].replace(name_mapper)

    return df


def _create_model_scatter_plot(
    df: pd.DataFrame, y_variable: str, y_variable_predicted: str, block=None
):
    """
    Takes a dataframe of predictions from a regression model, as well as the names of y_variable and
    y_variable_predicted. The function creates a scatter of the predicted values against actual values (with
    predicted values always on the x axis).

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of model predictions.
    y_variable : str
        Output variable for the model.
    y_variable_predicted : str
        Predictions for y_variable.
    block : bool, optional
        Block parameter for matplotlib.pyplot.show(). Whether to wait for all figures to be closed before returning.
        If True, block and run the GUI main loop until all figure windows are closed.
        If False, ensure that all figure windows are displayed and return immediately. In this case, you are responsible
        for ensuring that the event loop is running to have responsive figures. For more information see the
        `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html>`_.

    Returns
    -------
    plt
        A scatter plot of actual vs. predicted values.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> from prospect.models.plotting import create_model_predictions_scatter_plot
        >>> data = {
        ...     "y_actual": [3, 2, 4, 5, 6],
        ...     "y_predicted": [2.5, 2.1, 4.2, 4.8, 5.5],
        ... }
        >>> df = pd.DataFrame(data)
        >>> plot = create_model_predictions_scatter_plot(
        ...     df, "y_actual", "y_predicted"
        ... )  # doctest: +SKIP
    """

    # Raise an error if y_variable or y_variable_predicted are not in the dataframe
    if y_variable not in df.columns:
        raise Exception(f"ERROR: y_variable {y_variable} is not in the dataframe.")
    elif y_variable_predicted not in df.columns:
        raise Exception(
            f"ERROR: y_variable_predicted {y_variable_predicted} is not in the dataframe."
        )

    x = df[y_variable_predicted]
    y = df[y_variable]

    # Plot the scatter chart
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(x, y, alpha=1, color="#22D081")

    # Annotate the bar chart
    ax.set_ylabel(y_variable, size=12)
    ax.set_xlabel(y_variable_predicted, size=12)
    ax.set_title(f"{y_variable} vs {y_variable_predicted}", size=14, weight="bold")

    # compute slope for regression through origin
    beta = np.sum(x * y) / np.sum(x**2)

    # predicted values
    y_pred = beta * x

    # compute R² with intercept = 0
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum(y**2)  # Note: no centering since no intercept

    r2 = 1 - ss_res / ss_tot

    # Add the line of best fit
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color="black")
    plt.annotate(
        "r-squared = {:.3f}".format(r2),
        (min(x), max(y)),
        weight="bold",
    )

    # Show the plot
    plt.tight_layout()
    plt.show(block=block)

    return fig


def _create_jeopardy_heatmap_data(df, jeopardy_name, matches_per_bucket=9):
    """Creates a pivoted dataframe for heatmap visualization of jeopardy over matchweeks.

    This function groups matches into buckets (fixtures) and creates a matrix where
    rows represent match positions within each bucket and columns represent matchweeks.

    Args:
        df: A pandas DataFrame containing match data with columns including the
            specified jeopardy metric.
        jeopardy_name: The name of the jeopardy column to use for values
            (e.g., 'match_jeopardy_bottom_3', 'match_jeopardy_top_1').
        matches_per_bucket: The number of matches to group together into each
            fixture bucket. Defaults to 8.

    Returns:
        A pandas DataFrame with fixture labels as index (Match_1, Match_2, etc.)
            and matchweek labels as columns (Matchweek_1, Matchweek_2, etc.),
            containing jeopardy values.

    Example:
        >>> df = pd.DataFrame({"match_jeopardy_bottom_3": [0.1, 0.2, 0.3, 0.4]})
        >>> result = create_jeopardy_heatmap_data(df, "match_jeopardy_bottom_3", 2)
        >>> print(result)
                  Matchweek_1  Matchweek_2
        Match_1          0.1          0.3
        Match_2          0.2          0.4
    """
    # Create a copy to avoid modifying original dataframe
    df_copy = df.copy()

    # Calculate fixture (bucket) and position within bucket
    df_copy["fixture"] = (df_copy.index // matches_per_bucket) + 1
    df_copy["match_position"] = (df_copy.index % matches_per_bucket) + 1

    # Create match labels
    df_copy["match_label"] = "Match_" + df_copy["match_position"].astype(str)

    # Create matchweek labels
    df_copy["matchweek_label"] = "Matchweek_" + df_copy["fixture"].astype(str)

    # Pivot the dataframe
    heatmap_df = df_copy.pivot(
        index="match_label", columns="matchweek_label", values=jeopardy_name
    )

    # Sort index and columns to ensure proper ordering
    match_order = [f"Match_{i}" for i in range(1, matches_per_bucket + 1)]
    heatmap_df = heatmap_df.reindex(match_order)

    # Sort columns numerically
    matchweek_cols = sorted(heatmap_df.columns, key=lambda x: int(x.split("_")[1]))
    heatmap_df = heatmap_df[matchweek_cols]

    return heatmap_df


def _plot_jeopardy_heatmap(heatmap_df, jeopardy_name, figsize=(14, 6), cmap="Greens"):
    """Creates a heatmap visualization of jeopardy values across fixtures and matchweeks.

    Args:
        heatmap_df: A pandas DataFrame with fixture labels as index and matchweek
            labels as columns, containing jeopardy values.
        jeopardy_name: The name of the jeopardy metric being plotted, used for
            the title and colorbar label.
        figsize: A tuple specifying the figure size (width, height) in inches.
            Defaults to (14, 6).
        cmap: The colormap to use for the heatmap. Defaults to 'Greens' (white to green).

    Returns:
        A matplotlib Figure object containing the heatmap visualization.

    Example:
        >>> df = pd.DataFrame({"Matchweek_1": [0.1, 0.2]}, index=["Match_1", "Match_2"])
        >>> fig = plot_jeopardy_heatmap(df, "match_jeopardy_bottom_3")
        >>> fig.savefig("heatmap.png")
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract numbers from index and columns
    match_numbers = [int(idx.split("_")[1]) for idx in heatmap_df.index]
    matchweek_numbers = [int(col.split("_")[1]) for col in heatmap_df.columns]

    # Create heatmap with fixed scale from 0 to 0.5
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": jeopardy_name},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=0.5,
    )

    # Set tick labels to just the numbers
    ax.set_xticklabels(matchweek_numbers)
    ax.set_yticklabels(match_numbers)

    # Set labels and title
    ax.set_xlabel("Matchweek", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fixture", fontsize=12, fontweight="bold")

    # Format jeopardy name for title
    title_name = jeopardy_name.replace("_", " ").title()
    ax.set_title(
        f"Heatmap of {title_name} Over Time - 2024/2025",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Rotate labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()

    return fig


def create_model_df(
    attendance_df: pd.DataFrame,
    player_elo_df: pd.DataFrame,
    prospect_jeopardy_df: pd.DataFrame,
    params: dict,
):
    """
    Builds a feature-enriched model DataFrame for match attendance prediction by
    merging attendance data with player and team Elo ratings, engineering matchup-based
    features, and applying schema validation.

    The function performs the following steps:
    - Cleans raw attendance data.
    - Aggregates player Elo ratings into team-level pre-match ratings.
    - Merges home and away team ratings onto match records using as-of joins with date
      tolerance.
    - Computes combined and differential rating features.
    - Extracts additional temporal features (e.g. year, month, weekday).
    - Adds time slot categorization and derby indicators.
    - Validates the final dataset against a predefined schema.

    Args:
        attendance_df (pd.DataFrame):
            Raw attendance data containing match-level information including dates and teams.
        player_elo_df (pd.DataFrame):
            Player Elo ratings over time, used to compute team-level performance metrics.
        name_mapper (dict):
            Mapping from raw team names in `player_elo_df` to standardized names matching
            `attendance_df`.
        derbies_mapper (dict):
            Mapping of known derby matchups used to flag rivalry fixtures.

    Returns:
        pd.DataFrame:
            A fully processed and validated DataFrame ready for modeling, with one row per match
            containing attendance, team ratings, engineered features, and contextual attributes.
    """

    clean_attendance_data_df = _clean_attendance_data(attendance_df)

    player_and_team_df, latest_team_ratings_df = _aggregate_player_and_team_data(
        player_elo_df
    )

    clean_jeopardy_df = _get_latest_leverage_simulations(
        prospect_jeopardy_df[prospect_jeopardy_df["season"] != "2025/2026"],
        ["top_1", "top_8", "bottom_2", "bottom_3"],
    )

    clean_jeopardy_df["match_jeopardy_play_offs"] = clean_jeopardy_df[
        "match_jeopardy_top_8"
    ].fillna(0)

    clean_jeopardy_df["match_jeopardy_relegation"] = clean_jeopardy_df[
        "match_jeopardy_bottom_2"
    ].fillna(0) + clean_jeopardy_df["match_jeopardy_bottom_3"].fillna(0)

    clean_jeopardy_df = clean_jeopardy_df.rename(
        columns={
            "match_jeopardy_top_1": "match_jeopardy_title",
        }
    )

    prospect_performance_metrics_df = _create_single_prospect_df(
        clean_jeopardy_df,
        player_and_team_df,
        params["team_elo_to_simulations_match_id_mapper"],
    )

    # Apply name mapping
    prospect_performance_metrics_df["home_team_name"] = prospect_performance_metrics_df[
        "home_team_name"
    ].replace(params["team_name_mapping_dict"])

    prospect_performance_metrics_df["away_team_name"] = prospect_performance_metrics_df[
        "away_team_name"
    ].replace(params["team_name_mapping_dict"])

    model_df = clean_attendance_data_df.merge(
        prospect_performance_metrics_df[
            [
                "season",
                "home_team_name",
                "away_team_name",
                "prospect_home_team_rating",
                "prospect_home_squad_rating",
                "prospect_away_team_rating",
                "prospect_away_squad_rating",
                "prospect_combined_team_rating",
                "prospect_combined_squad_rating",
                "prospect_team_rating_abs_difference",
                "home_team_win_prob",
                "match_jeopardy_title",
                "match_jeopardy_relegation",
                "match_jeopardy_play_offs",
            ]
        ],
        left_on=["season", "home_team", "away_team"],
        right_on=["season", "home_team_name", "away_team_name"],
        how="left",
    )

    model_df = _extract_date_features(model_df, "match_date")

    model_df = _add_in_time_slots(model_df)

    model_df = _add_in_derbies(model_df, params["team_derbies_dict"])

    validated_attendance_model_df = _validate_with_schema(
        model_df, attendance_model_schema
    )

    # ensures there is no bad data as occasionally raw attendance data exceeds venue capacity which cannot happen
    validated_attendance_model_df["venue_percentage_capacity"] = (
        validated_attendance_model_df["venue_percentage_capacity"].clip(upper=1)
    )

    return validated_attendance_model_df, latest_team_ratings_df


def create_viewership_model_df(
    viewership_data_sheet_1: pd.DataFrame,
    viewership_data_sheet_2: pd.DataFrame,
    attendance_model_df: pd.DataFrame,
    name_mapper: dict,
) -> pd.DataFrame:
    """Creates a viewership model DataFrame by cleaning, merging, and mapping viewership data
    with attendance data.

    Args:
        viewership_data_sheet_1 (pd.DataFrame): First raw viewership data sheet (e.g., current season).
        viewership_data_sheet_2 (pd.DataFrame): Second raw viewership data sheet (e.g., previous season).
        attendance_model_df (pd.DataFrame): DataFrame containing attendance information.
        name_mapper (dict): Dictionary mapping domestic team names to standardized names.

    Returns:
        pd.DataFrame: A DataFrame containing merged viewership and attendance data for domestic matches.
    """

    # Add season columns
    viewership_data_sheet_1["season"] = "2024/2025"
    viewership_data_sheet_2["season"] = "2023/2024"

    # Combine and clean international and domestic viewership
    international_viewership_df = pd.concat(
        [
            viewership_data_sheet_1[["season", "Unnamed: 8", "Unnamed: 9"]],
            viewership_data_sheet_2[["season", "Unnamed: 8", "Unnamed: 9"]],
        ]
    )
    domestic_viewership_df = pd.concat(
        [
            viewership_data_sheet_1[["season", "Unnamed: 1", "Unnamed: 2"]],
            viewership_data_sheet_2[["season", "Unnamed: 1", "Unnamed: 2"]],
        ]
    )

    international_viewership_df = _clean_viewership_df(
        international_viewership_df,
        match_col="Unnamed: 8",
        viewership_col="Unnamed: 9",
        name_mapper=name_mapper,
    )
    domestic_viewership_df = _clean_viewership_df(
        domestic_viewership_df,
        match_col="Unnamed: 1",
        viewership_col="Unnamed: 2",
        name_mapper=name_mapper,
    )

    # Filter attendance_model_df for relevant seasons
    attendance_model_df = attendance_model_df[
        attendance_model_df["season"].isin(["2023/2024", "2024/2025"])
    ].reset_index(drop=True)

    # Merge domestic viewership with attendance
    domestic_viewership_model_df = attendance_model_df.merge(
        domestic_viewership_df,
        on=["season", "home_team", "away_team"],
        how="right",
    )

    international_viewership_model_df = attendance_model_df.merge(
        international_viewership_df,
        on=["season", "home_team", "away_team"],
        how="right",
    )

    validated_domestic_viewership_model_df = _validate_with_schema(
        domestic_viewership_model_df, viewership_model_schema
    )

    validated_international_viewership_model_df = _validate_with_schema(
        international_viewership_model_df, viewership_model_schema
    )

    return (
        validated_domestic_viewership_model_df,
        validated_international_viewership_model_df,
    )


def scatter_plot_of_team_rating_vs_squad_rating(
    df: pd.DataFrame,
) -> plt.Figure:
    """Create a scatter plot comparing team ratings against squad ratings.

    This function combines home and away team/squad ratings from a match DataFrame
    into a single dataset and generates a scatter plot to visualize the relationship
    between team ratings and squad ratings across all teams.

    Args:
        df: Input DataFrame containing match data with the following columns:
            - prospect_home_team_rating: Team rating for home team
            - prospect_away_team_rating: Team rating for away team
            - prospect_home_squad_rating: Squad rating for home team
            - prospect_away_squad_rating: Squad rating for away team

    Returns:
        Matplotlib Figure object containing the scatter plot visualization.

    Examples:
        >>> match_data = pd.DataFrame(
        ...     {
        ...         "prospect_home_team_rating": [75, 80, 85],
        ...         "prospect_away_team_rating": [70, 78, 82],
        ...         "prospect_home_squad_rating": [72, 79, 83],
        ...         "prospect_away_squad_rating": [68, 76, 80],
        ...     }
        ... )
        >>> fig = scatter_plot_of_team_rating_vs_squad_rating(match_data)

    Notes:
        - Each match contributes two data points (home and away teams)
        - A DataFrame with N matches will produce 2N points in the scatter plot
        - The plot is displayed with block=False, allowing continued execution
        - Useful for analyzing correlation between team and squad ratings
    """
    # Combine home and away ratings into a single dataset
    team_df = pd.DataFrame(
        {
            "prospect_team_rating": df["prospect_home_team_rating"].tolist()
            + df["prospect_away_team_rating"].tolist(),
            "prospect_squad_rating": df["prospect_home_squad_rating"].tolist()
            + df["prospect_away_squad_rating"].tolist(),
        }
    )

    # Create scatter plot using helper function
    plot = _create_model_scatter_plot(
        team_df, "prospect_team_rating", "prospect_squad_rating", block=False
    )

    return plot


def heatmap_of_jeopardy_over_time(
    jeopardy_df: pd.DataFrame,
    season: str = "2024/2025",
    jeopardy_types: list[str] = None,
) -> dict[str, tuple[pd.DataFrame, plt.Figure]]:
    """Creates heatmap visualizations of match jeopardy metrics over time.

    This function processes jeopardy data for a specific season, creates pivoted
    dataframes for heatmap visualization, and generates heatmap plots for multiple
    jeopardy metrics (relegation, title, and playoff positions).

    Args:
        jeopardy_df: A pandas DataFrame containing match jeopardy data with columns
            including 'season', 'match_jeopardy_top_1', 'match_jeopardy_top_8', and
            'match_jeopardy_bottom_3'.
        season: The season to filter data for. Defaults to "2024/2025".
        jeopardy_types: A list of jeopardy types to include. If None, defaults to
            ["top_1", "top_8", "bottom_3"].

    Returns:
        A dictionary mapping jeopardy metric names to tuples of (dataframe, figure).
        Keys are: 'relegation', 'title', and 'playoff'.
        Each tuple contains the pivoted heatmap dataframe and the corresponding
        matplotlib Figure object.

    Example:
        >>> results = heatmap_of_jeopardy_over_time(df)
        >>> relegation_df, relegation_fig = results["relegation"]
        >>> relegation_fig.savefig("relegation_heatmap.png")
        >>> title_df, title_fig = results["title"]
    """
    if jeopardy_types is None:
        jeopardy_types = ["top_1", "top_8", "bottom_3"]

    # Filter and clean the data
    clean_jeopardy_df = _get_latest_leverage_simulations(
        jeopardy_df[jeopardy_df["season"] == season],
        jeopardy_types,
    )

    # Rename columns for clarity
    clean_jeopardy_df = clean_jeopardy_df.rename(
        columns={
            "match_jeopardy_top_1": "match_jeopardy_title",
            "match_jeopardy_bottom_3": "match_jeopardy_relegation",
            "match_jeopardy_top_8": "match_jeopardy_playoff",
        }
    )

    # Define the jeopardy metrics to process
    jeopardy_metrics = {
        "Relegation": "match_jeopardy_relegation",
        "Title": "match_jeopardy_title",
        "Playoff": "match_jeopardy_playoff",
    }

    # Create heatmaps for all metrics
    dataframes_dict = {}
    plots = []

    for metric_name, column_name in jeopardy_metrics.items():
        heatmap_df = _create_jeopardy_heatmap_data(clean_jeopardy_df, column_name)
        heatmap_plot = _plot_jeopardy_heatmap(heatmap_df, column_name)

        # Store dataframe with sheet name
        dataframes_dict[metric_name] = heatmap_df.reset_index()

        # Store plot in order: relegation, title, playoff
        plots.append(heatmap_plot)

    # Return dictionary and plots
    return dataframes_dict, plots[0], plots[1], plots[2]
