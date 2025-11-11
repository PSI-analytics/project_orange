"""
This is a boilerplate pipeline 'model_predictions'
generated using Kedro 1.0.0
"""

import numpy as np
import pandas as pd
from prospect.models.predictions import (
    create_model_ensemble_predictions_df,
)

from ..data_schemas.data_schemas import (
    attendance_model_schema,
    viewership_model_schema,
)
from ..model_generation.nodes import (
    _convert_model_df_into_one_hot_encoded_model_df,
)
from ..model_preprocessing.nodes import (
    _validate_with_schema,
    _impute_below_season_threshold,
    _get_latest_leverage_simulations,
    _process_jeopardy_dataframe,
)


def _make_stacked_model_prediction(
    attendance_model_df: pd.DataFrame,
    predicted_attendance_linear_model: dict,
    predicted_attendance_linear_model_scaler,
    predicted_attendance_linear_model_params: dict,
    predicted_attendance_xgboost_model: dict,
    predicted_attendance_xgboost_model_scaler,
    predicted_attendance_xgboost_model_params: dict,
    schema,
):
    """
    Generates ensemble-based attendance predictions using a stacked linear and XGBoost model.

    This function performs the following steps:
    1. Validates the input DataFrame schema.
    2. Applies scaling to numerical columns if required.
    3. One-hot encodes categorical variables for model ingestion.
    4. Generates predictions from the base linear model.
    5. Feeds linear model predictions into the XGBoost model for final stacking.
    6. Converts predicted venue percentage capacity back into attendance counts.

    Args:
        attendance_model_df (pd.DataFrame): Input feature DataFrame containing match
            and venue attributes. Expected to conform to `attendance_model_schema`.
        predicted_attendance_linear_model (dict): Trained linear model object used for
            generating the first-layer predictions.
        predicted_attendance_linear_model_scaler: Scaler used to normalize numerical
            columns prior to inference (e.g., StandardScaler or MinMaxScaler).
        predicted_attendance_linear_model_params (dict): Metadata dictionary containing
            model configuration such as:
                - "standardise_data" (bool): Whether to scale numeric input features.
                - "numerical_columns" (list[str]): Columns to be scaled.
                - "model_type" (str): Identifier for the model (e.g., "linear").
                - "target_variable" (str): Column name of the target variable.
        predicted_attendance_xgboost_model (dict): Trained XGBoost model used for
            stacked second-layer predictions.
        predicted_attendance_xgboost_model_scaler: Scaler used internally for XGBoost input
            preprocessing (unused directly here but kept for consistency).
        predicted_attendance_xgboost_model_params (dict): Configuration for the XGBoost
            model with the same structure as the linear model params.

    Returns:
        pd.DataFrame: Original validated DataFrame with additional prediction columns for
        each model stage, including both percentage-based and absolute attendance estimates.

    Raises:
        pa.errors.SchemaError: If the input DataFrame does not conform to the required schema.
    """

    validated_attendance_model_df = _validate_with_schema(attendance_model_df, schema)

    scaled_model_df = validated_attendance_model_df.copy()
    if predicted_attendance_linear_model_params["standardise_data"]:
        model_df_scaled_df = scaled_model_df[
            predicted_attendance_linear_model_params["numerical_columns"]
        ]
        model_df_scaled_df = predicted_attendance_linear_model_scaler.transform(
            model_df_scaled_df
        )
        scaled_model_df[
            predicted_attendance_linear_model_params["numerical_columns"]
        ] = model_df_scaled_df

    (
        one_hot_encoded_model_df,
        model_columns,
    ) = _convert_model_df_into_one_hot_encoded_model_df(
        scaled_model_df, predicted_attendance_linear_model_params
    )

    model_predictions_df, _ = create_model_ensemble_predictions_df(
        one_hot_encoded_model_df,
        predicted_attendance_linear_model,
        return_individual_predictions_df=False,
    )

    # rename linear model prediction
    model_predictions_df = model_predictions_df.rename(
        columns={
            predicted_attendance_linear_model_params["target_variable"]
            + "_predicted": predicted_attendance_linear_model_params["target_variable"]
            + "_"
            + predicted_attendance_linear_model_params["model_type"]
            + "_predicted"
        }
    )

    (
        all_regression_predicted_earnings_model_ensemble_predictions_df,
        _,
    ) = create_model_ensemble_predictions_df(
        model_predictions_df,
        predicted_attendance_xgboost_model,
        return_individual_predictions_df=False,
        rename_predicted_columns_dict={},
    )

    validated_attendance_model_df[
        predicted_attendance_linear_model_params["target_variable"]
        + "_"
        + predicted_attendance_linear_model_params["model_type"]
        + "_predicted"
    ] = model_predictions_df[
        predicted_attendance_linear_model_params["target_variable"]
        + "_"
        + predicted_attendance_linear_model_params["model_type"]
        + "_predicted"
    ]

    validated_attendance_model_df[
        predicted_attendance_xgboost_model_params["target_variable"]
        + "_"
        + predicted_attendance_xgboost_model_params["model_type"]
        + "_predicted"
    ] = all_regression_predicted_earnings_model_ensemble_predictions_df[
        predicted_attendance_xgboost_model_params["target_variable"] + "_predicted"
    ]

    # predict attendance for linear model
    if (
        predicted_attendance_linear_model_params["target_variable"]
        == "venue_percentage_capacity"
    ):
        validated_attendance_model_df[
            "attendance_"
            + predicted_attendance_linear_model_params["model_type"]
            + "_predicted"
        ] = (
            validated_attendance_model_df[
                predicted_attendance_linear_model_params["target_variable"]
                + "_"
                + predicted_attendance_linear_model_params["model_type"]
                + "_predicted"
            ]
            * validated_attendance_model_df["venue_capacity"]
        )

    # predict attendance for xgboost model
    if (
        predicted_attendance_xgboost_model_params["target_variable"]
        == "venue_percentage_capacity"
    ):
        validated_attendance_model_df[
            "attendance_"
            + predicted_attendance_xgboost_model_params["model_type"]
            + "_predicted"
        ] = (
            validated_attendance_model_df[
                predicted_attendance_xgboost_model_params["target_variable"]
                + "_"
                + predicted_attendance_xgboost_model_params["model_type"]
                + "_predicted"
            ]
            * validated_attendance_model_df["venue_capacity"]
        )

    return validated_attendance_model_df


def _add_playoff_fixtures(
    df: pd.DataFrame,
    n_fixtures: int = 7,
    fixed_params: dict = None,
) -> pd.DataFrame:
    """Add playoff fixtures to a football DataFrame with fixed and averaged values.

    This function appends new rows to an existing football match DataFrame.
    Columns defined in `fixed_params` are filled with specified fixed values.
    For all other numeric columns, the function fills them with the column average
    computed from the existing dataset. Non-numeric and non-fixed columns are set to NaN.

    Args:
        df (pd.DataFrame): The original match dataset.
        n_fixtures (int, optional): Number of playoff fixture rows to add. Defaults to 7.
        fixed_params (dict, optional): A dictionary mapping column names to fixed values.
            Example:
                {
                    "season": "2024/2025",
                    "month": "May",
                    "time_slot": "Saturday 21:00",
                    "derby": 0,
                    "match_jeopardy_title": 1,
                    "match_jeopardy_relegation": 0,
                    "venue_name": "King Abdullah Sport City",
                    "venue_capacity": 60348
                }

    Returns:
        pd.DataFrame: The updated DataFrame with the added playoff fixtures.
    """
    # Default fixed parameters
    if fixed_params is None:
        fixed_params = {
            "season": "2024/2025",
            "month": "May",
            "time_slot": "Saturday 21:00",
            "derby": 0,
            "match_jeopardy_title": 1,
            "match_jeopardy_relegation": 0,
            "venue_name": "King Abdullah Sport City",
            "venue_capacity": 60348,
        }

    # Compute column means for all numeric columns
    averages = df.mean(numeric_only=True)

    new_rows = []
    for _ in range(n_fixtures):
        row = {}
        for col in df.columns:
            if col in fixed_params:
                row[col] = fixed_params[col]
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Use average if numeric
                row[col] = averages.get(col, np.nan)
            else:
                row[col] = np.nan
        new_rows.append(row)

    playoff_df = pd.DataFrame(new_rows, columns=df.columns)

    playoff_df = _add_combined_ratings(playoff_df)

    playoff_df["total_match_jeopardy"] = 1

    # Concatenate original with playoff fixtures
    df_extended = pd.concat([df, playoff_df], ignore_index=True)

    return df_extended


def _create_jeopardy_scenarios(
    model_df: pd.DataFrame,
    jeopardy_df: pd.DataFrame,
    commercial_simulations_params: dict,
    playoff_impact_factor: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create multiple jeopardy scenarios for model evaluation.

    This function generates four different scenarios by manipulating jeopardy-related
    columns to test different assumptions about match importance:
    - Baseline: Original data unchanged
    - Scenario 1: No relegation jeopardy
    - Scenario 2: Replace title jeopardy with scaled playoff jeopardy
    - Scenario 3: No relegation jeopardy + replace title jeopardy with scaled playoff jeopardy

    Args:
        model_df: A pandas DataFrame containing match data with the following columns:
            - match_jeopardy_relegation: Relegation jeopardy metric
            - match_jeopardy_title: Title jeopardy metric
            - match_jeopardy_play_offs: Playoff jeopardy metric
        playoff_impact_factor: Multiplicative factor to scale playoff jeopardy when
            substituting for title jeopardy. Default is 1.0 (100% impact). A value
            of 0.5 would represent 50% impact.

    Returns:
        A tuple containing four DataFrames:
            - baseline_df: Original model_df unchanged
            - scenario_1_df: Relegation jeopardy set to 0
            - scenario_2_df: Title jeopardy replaced with scaled playoff jeopardy
            - scenario_3_df: Both modifications applied (no relegation + scaled playoff)

    Raises:
        KeyError: If required columns are missing from model_df.

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "match_jeopardy_relegation": [1, 2, 3],
        ...         "match_jeopardy_title": [4, 5, 6],
        ...         "match_jeopardy_play_offs": [7, 8, 9],
        ...     }
        ... )
        >>> baseline, s1, s2, s3 = _create_jeopardy_scenarios(
        ...     df, playoff_impact_factor=0.5
        ... )
        >>> s2["match_jeopardy_title"].tolist()
        [3.5, 4.0, 4.5]
    """
    # Validate required columns exist
    required_columns = [
        "match_jeopardy_relegation",
        "match_jeopardy_title",
        "match_jeopardy_play_offs",
    ]
    missing_columns = [col for col in required_columns if col not in model_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    model_df = model_df.drop(
        columns=[
            "match_jeopardy_relegation",
            "match_jeopardy_title",
            "match_jeopardy_play_offs",
        ]
    )

    # replace with future jeopardy from the simulators
    model_df = model_df.merge(
        jeopardy_df[
            [
                "season",
                "home_team",
                "away_team",
                "match_jeopardy_relegation",
                "match_jeopardy_title",
                "match_jeopardy_play_offs",
            ]
        ],
        on=["season", "home_team", "away_team"],
        how="left",
    )

    # fill na with 0 for missing seasons. Keep previous seasons as logic for one hot encoding when making model predictions
    # requires it
    model_df[
        [
            "match_jeopardy_relegation",
            "match_jeopardy_title",
            "match_jeopardy_play_offs",
        ]
    ] = model_df[
        [
            "match_jeopardy_relegation",
            "match_jeopardy_title",
            "match_jeopardy_play_offs",
        ]
    ].fillna(0)

    # Baseline scenario: return original data unchanged
    baseline_df = model_df.copy()
    baseline_df["total_match_jeopardy"] = (
        baseline_df["match_jeopardy_relegation"] + baseline_df["match_jeopardy_title"]
    )

    # Scenario 1: Remove relegation jeopardy
    scenario_1_df = model_df.copy()
    scenario_1_df["match_jeopardy_relegation"] = 0
    scenario_1_df["total_match_jeopardy"] = (
        scenario_1_df["match_jeopardy_relegation"]
        + scenario_1_df["match_jeopardy_title"]
    )

    # Scenario 2: Replace title jeopardy with scaled playoff jeopardy
    scenario_2_df = model_df.copy()
    scenario_2_df["match_jeopardy_title"] = (
        scenario_2_df["match_jeopardy_play_offs"] * playoff_impact_factor
    )
    scenario_2_df["total_match_jeopardy"] = (
        scenario_2_df["match_jeopardy_relegation"]
        + scenario_2_df["match_jeopardy_play_offs"]
    )

    scenario_2_df = _add_playoff_fixtures(
        scenario_2_df,
        commercial_simulations_params["number_of_playoff_fixtures"],
        commercial_simulations_params["playoff_variables"],
    )

    # Scenario 3: Combine both modifications
    scenario_3_df = model_df.copy()
    scenario_3_df["match_jeopardy_relegation"] = 0
    scenario_3_df["match_jeopardy_title"] = (
        scenario_3_df["match_jeopardy_play_offs"] * playoff_impact_factor
    )
    scenario_3_df["total_match_jeopardy"] = (
        scenario_3_df["match_jeopardy_relegation"]
        + scenario_3_df["match_jeopardy_play_offs"]
    )

    scenario_3_df = _add_playoff_fixtures(
        scenario_3_df,
        commercial_simulations_params["number_of_playoff_fixtures"],
        commercial_simulations_params["playoff_variables"],
    )

    return baseline_df, scenario_1_df, scenario_2_df, scenario_3_df


def _run_model_scenarios(
    model_df: pd.DataFrame,
    future_jeopardy_df: pd.DataFrame,
    linear_model: dict,
    linear_scaler,
    linear_params: dict,
    xgb_model: dict,
    xgb_scaler,
    xgb_params: dict,
    schema,
    commercial_simulations_params: dict,
    season_filter: str = "2024/2025",
) -> dict[str, pd.DataFrame]:
    """Run predictions across multiple uplift and jeopardy scenarios.

    This function creates a comprehensive set of prediction scenarios by combining:
    - Three uplift scenarios (status quo, average uplift, top teams uplift)
    - Four jeopardy scenarios (baseline, no relegation, playoff impact, combined)

    This results in 12 total scenario combinations (3 x 4), each with predictions
    from a stacked model (linear + XGBoost).

    Args:
        model_df: Base DataFrame containing match data with features for prediction.
        linear_model: Trained linear model object or parameters for prediction.
        linear_scaler: Fitted scaler for linear model feature preprocessing.
        linear_params: Configuration parameters for the linear model.
        xgb_model: Trained XGBoost model object or parameters for prediction.
        xgb_scaler: Fitted scaler for XGBoost feature preprocessing.
        xgb_params: Configuration parameters for the XGBoost model.
        schema: Schema definition used by `_make_stacked_model_prediction` for
            feature selection and validation.
        season_filter: Season string to filter results. Defaults to "2024/2025".

    Returns:
        Dictionary mapping scenario names to prediction DataFrames. Keys follow the
        pattern "{uplift_type}_scenario_{jeopardy_number}" where:
            - uplift_type: "status_quo", "uplift_average", or "uplift_top_teams"
            - jeopardy_number: 1-4 representing different jeopardy configurations

        Example keys: "status_quo_scenario_1", "uplift_average_scenario_3"

    Examples:
        >>> scenarios = _run_model_scenarios(
        ...     model_df=match_data,
        ...     linear_model=lr_model,
        ...     linear_scaler=lr_scaler,
        ...     linear_params=lr_params,
        ...     xgb_model=xgb_model,
        ...     xgb_scaler=xgb_scaler,
        ...     xgb_params=xgb_params,
        ...     schema=feature_schema,
        ... )
        >>> list(scenarios.keys())
        ['status_quo_scenario_1', 'status_quo_scenario_2', ..., 'uplift_top_teams_scenario_4']

    Notes:
        - All returned DataFrames are filtered to the specified season
        - The function creates three base uplift scenarios before applying jeopardy variations
        - Status quo: No modifications to squad/team ratings
        - Average uplift: Imputes below-threshold ratings with season averages
        - Top teams uplift: Imputes using specific top-performing clubs (Al Hilal, Al Nassr)
    """
    # Rating columns used for uplift calculations
    modified_features_list = [
        "prospect_home_squad_rating",
        "prospect_away_squad_rating",
        "prospect_home_team_rating",
        "prospect_away_team_rating",
    ]

    # Define uplift scenarios
    uplift_scenarios = _create_uplift_scenarios(model_df, modified_features_list)

    # Generate jeopardy scenarios for each uplift scenario
    jeopardy_scenarios = {}
    for uplift_name, uplift_df in uplift_scenarios.items():
        jeopardy_dfs = _create_jeopardy_scenarios(
            uplift_df,
            future_jeopardy_df[
                future_jeopardy_df["scenario"] == uplift_name
            ].reset_index(drop=True),
            commercial_simulations_params,
            0.5,
        )
        for jeopardy_idx, jeopardy_df in enumerate(jeopardy_dfs, start=1):
            scenario_key = f"{uplift_name}_scenario_{jeopardy_idx}"
            jeopardy_scenarios[scenario_key] = jeopardy_df

    # Run predictions for all scenarios
    prediction_scenarios = {}
    for scenario_name, scenario_df in jeopardy_scenarios.items():
        prediction_scenarios[scenario_name] = _make_stacked_model_prediction(
            scenario_df,
            linear_model,
            linear_scaler,
            linear_params,
            xgb_model,
            xgb_scaler,
            xgb_params,
            schema,
        )

        # split out regular season and play-offs
        play_off_df = prediction_scenarios[scenario_name][
            prediction_scenarios[scenario_name]["match_jeopardy_title"] == 1
        ]

        if len(play_off_df) > 0:
            prediction_scenarios[scenario_name] = prediction_scenarios[scenario_name][
                prediction_scenarios[scenario_name]["match_jeopardy_title"] != 1
            ].reset_index(drop=True)
            prediction_scenarios[scenario_name + "- playoffs"] = play_off_df

            # Filter all scenarios to the target season
    filtered_scenarios = {
        key: df[df["season"] == season_filter]
        for key, df in prediction_scenarios.items()
    }

    return filtered_scenarios


def _create_uplift_scenarios(
    model_df: pd.DataFrame,
    rating_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Create three uplift scenarios with different rating imputation strategies.

    Args:
        model_df: Base DataFrame containing match and rating data.
        rating_columns: List of rating column names to impute.

    Returns:
        Dictionary with three uplift scenarios:
            - "status_quo": Original data with combined ratings calculated
            - "uplift_average": Ratings imputed with season averages + combined ratings
            - "uplift_top_teams": Ratings imputed from top clubs + combined ratings
    """
    # Status quo: Use original data
    status_quo = model_df.copy()
    status_quo = _add_combined_ratings(status_quo)

    # Average uplift: Impute with season averages
    uplift_average = _impute_below_season_threshold(model_df, rating_columns)
    uplift_average = _add_combined_ratings(uplift_average)

    # Top teams uplift: Impute using specific top-performing clubs
    uplift_top_teams = _impute_below_season_threshold(
        model_df,
        rating_columns,
        method="club",
        club_name_list=["Al Hilal", "Al Nassr"],
    )
    uplift_top_teams = _add_combined_ratings(uplift_top_teams)

    return {
        "status_quo": status_quo,
        "uplift_to_average_squad": uplift_average,
        "uplift_to_top_teams": uplift_top_teams,
    }


def _add_combined_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined squad and team rating columns.

    Args:
        df: DataFrame containing home and away squad/team ratings.

    Returns:
        DataFrame with added combined rating columns.
    """
    df["prospect_combined_squad_rating"] = (
        df["prospect_home_squad_rating"] + df["prospect_away_squad_rating"]
    )
    df["prospect_combined_team_rating"] = (
        df["prospect_home_team_rating"] + df["prospect_away_team_rating"]
    )
    df["prospect_team_rating_abs_difference"] = abs(
        df["prospect_home_team_rating"] - df["prospect_away_team_rating"]
    )
    return df


def run_commercial_scenarios(
    attendance_model_df: pd.DataFrame,
    domestic_viewership_model_df: pd.DataFrame,
    international_viewership_model_df: pd.DataFrame,
    future_jeopardy_df: pd.DataFrame,
    predicted_attendance_linear_model: dict,
    predicted_attendance_linear_model_scaler,
    predicted_attendance_linear_model_params: dict,
    predicted_attendance_xgboost_model: dict,
    predicted_attendance_xgboost_model_scaler,
    predicted_attendance_xgboost_model_params: dict,
    predicted_domestic_viewership_linear_model: dict,
    predicted_domestic_viewership_linear_model_scaler,
    predicted_domestic_viewership_linear_model_params: dict,
    predicted_domestic_viewership_xgboost_model: dict,
    predicted_domestic_viewership_xgboost_model_scaler,
    predicted_domestic_viewership_xgboost_model_params: dict,
    predicted_international_viewership_linear_model: dict,
    predicted_international_viewership_linear_model_scaler,
    predicted_international_viewership_linear_model_params: dict,
    predicted_international_viewership_xgboost_model: dict,
    predicted_international_viewership_xgboost_model_scaler,
    predicted_international_viewership_xgboost_model_params: dict,
    commercial_simulations_params: dict,
    future_jeopardy_scenario_1_df: pd.DataFrame,
    future_jeopardy_scenario_2_df: pd.DataFrame,
):
    """
    Runs commercial scenario predictions for attendance, domestic viewership,
    and international viewership models across three scenarios:
    status quo, uplift to average squad, and uplift to top quartile squad.

    This function:
    - Generates baseline predictions using the original model data.
    - Generates modified predictions for average and top quartile squad uplift scenarios.
    - Aggregates total predicted values for each scenario into a summary DataFrame.

    Args:
        attendance_model_df (pd.DataFrame): DataFrame containing features for attendance prediction.
        domestic_viewership_model_df (pd.DataFrame): DataFrame containing features for domestic viewership prediction.
        international_viewership_model_df (pd.DataFrame): DataFrame containing features for international viewership prediction.
        predicted_attendance_linear_model (dict): Linear regression model or parameters for attendance prediction.
        predicted_attendance_linear_model_scaler: Scaler for linear attendance model input features.
        predicted_attendance_linear_model_params (dict): Additional parameters for linear attendance model prediction.
        predicted_attendance_xgboost_model (dict): XGBoost model or parameters for attendance prediction.
        predicted_attendance_xgboost_model_scaler: Scaler for XGBoost attendance model input features.
        predicted_attendance_xgboost_model_params (dict): Additional parameters for XGBoost attendance model prediction.
        predicted_domestic_viewership_linear_model (dict): Linear regression model or parameters for domestic viewership prediction.
        predicted_domestic_viewership_linear_model_scaler: Scaler for linear domestic viewership model input features.
        predicted_domestic_viewership_linear_model_params (dict): Additional parameters for linear domestic viewership prediction.
        predicted_domestic_viewership_xgboost_model (dict): XGBoost model or parameters for domestic viewership prediction.
        predicted_domestic_viewership_xgboost_model_scaler: Scaler for XGBoost domestic viewership model input features.
        predicted_domestic_viewership_xgboost_model_params (dict): Additional parameters for XGBoost domestic viewership prediction.
        predicted_international_viewership_linear_model (dict): Linear regression model or parameters for international viewership prediction.
        predicted_international_viewership_linear_model_scaler: Scaler for linear international viewership model input features.
        predicted_international_viewership_linear_model_params (dict): Additional parameters for linear international viewership prediction.
        predicted_international_viewership_xgboost_model (dict): XGBoost model or parameters for international viewership prediction.
        predicted_international_viewership_xgboost_model_scaler: Scaler for XGBoost international viewership model input features.
        predicted_international_viewership_xgboost_model_params (dict): Additional parameters for XGBoost international viewership prediction.

    Returns:
        pd.DataFrame: A summary DataFrame with one row per scenario, containing aggregated predictions:
            - scenario (str): Scenario label ("status quo", "uplift to average squad", "uplift to top quartile squad").
            - total_attendance_linear_regression (float): Total predicted attendance using linear regression.
            - total_attendance_xgboost (float): Total predicted attendance using XGBoost.
            - total_domestic_viewership_linear_regression (float): Total predicted domestic viewership using linear regression.
            - total_domestic_viewership_xgboost (float): Total predicted domestic viewership using XGBoost.
            - total_international_viewership_linear_regression (float): Total predicted international viewership using linear regression.
            - total_international_viewership_xgboost (float): Total predicted international viewership using XGBoost.
    """
    # Run each scenario group
    attendance_preds = _run_model_scenarios(
        attendance_model_df,
        future_jeopardy_df,
        predicted_attendance_linear_model,
        predicted_attendance_linear_model_scaler,
        predicted_attendance_linear_model_params,
        predicted_attendance_xgboost_model,
        predicted_attendance_xgboost_model_scaler,
        predicted_attendance_xgboost_model_params,
        attendance_model_schema,
        commercial_simulations_params,
    )

    domestic_preds = _run_model_scenarios(
        domestic_viewership_model_df,
        future_jeopardy_df,
        predicted_domestic_viewership_linear_model,
        predicted_domestic_viewership_linear_model_scaler,
        predicted_domestic_viewership_linear_model_params,
        predicted_domestic_viewership_xgboost_model,
        predicted_domestic_viewership_xgboost_model_scaler,
        predicted_domestic_viewership_xgboost_model_params,
        viewership_model_schema,
        commercial_simulations_params,
    )

    international_preds = _run_model_scenarios(
        international_viewership_model_df,
        future_jeopardy_df,
        predicted_international_viewership_linear_model,
        predicted_international_viewership_linear_model_scaler,
        predicted_international_viewership_linear_model_params,
        predicted_international_viewership_xgboost_model,
        predicted_international_viewership_xgboost_model_scaler,
        predicted_international_viewership_xgboost_model_params,
        viewership_model_schema,
        commercial_simulations_params,
    )

    # Aggregate results for all scenarios
    scenario_labels = [
        "status quo",
        "status quo - no relegation",
        "status quo - 8 team play-off (regular season)",
        "status quo - 8 team play-off (playoffs)",
        "status quo - no relegation & 8 team play-off (regular season)",
        "status quo - no relegation & 8 team play-off (playoffs)",
        "uplift to average squad",
        "uplift to average squad - no relegation",
        "uplift to average squad - 8 team play-off (regular season)",
        "uplift to average squad - 8 team play-off (playoffs)",
        "uplift to average squad - no relegation & 8 team play-off (regular season)",
        "uplift to average squad - no relegation & 8 team play-off (playoffs)",
        "uplift to top teams",
        "uplift to top teams - no relegation",
        "uplift to top teams - 8 team play-off (regular season)",
        "uplift to top teams - 8 team play-off (playoffs)",
        "uplift to top teams - no relegation & 8 team play-off (regular season)",
        "uplift to top teams - no relegation & 8 team play-off (playoffs)",
    ]

    overall_impact_df = pd.DataFrame(
        {
            "Scenario": scenario_labels,
            "Average Attendance": [
                attendance_preds[k]["attendance_linear_regression_predicted"]
                .mean()
                .astype(int)
                for k in attendance_preds
            ],
            "Average Domestic Viewership": [
                domestic_preds[k]["viewership_linear_regression_predicted"]
                .mean()
                .astype(int)
                for k in domestic_preds
            ],
            "Average International Viewership": [
                international_preds[k]["viewership_linear_regression_predicted"]
                .mean()
                .astype(int)
                for k in international_preds
            ],
            "Number of Matches": [len(attendance_preds[k]) for k in attendance_preds],
            "Total In Season Jeopardy": [
                attendance_preds[k]["total_match_jeopardy"].sum().astype(float)
                for k in attendance_preds
            ],
        }
    )

    return overall_impact_df


def process_future_jeopardy_data(
    future_jeopardy_df_commercial_scenario_1: pd.DataFrame,
    future_jeopardy_df_commercial_scenario_2: pd.DataFrame,
    future_jeopardy_base_case_df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Process future Jeopardy match data for multiple commercial scenarios.

    This function cleans and processes three Jeopardy DataFrames: a base case and
    two commercial scenarios. It applies leverage simulations, processes match
    columns, adds a scenario label, and concatenates the results into a single DataFrame.

    Args:
        future_jeopardy_df_commercial_scenario_1 (pd.DataFrame):
            DataFrame for commercial scenario 1 containing future Jeopardy match data.
        future_jeopardy_df_commercial_scenario_2 (pd.DataFrame):
            DataFrame for commercial scenario 2 containing future Jeopardy match data.
        future_jeopardy_base_case_df (pd.DataFrame):
            Base case DataFrame containing future Jeopardy match data.

    Returns:
        pd.DataFrame: A concatenated DataFrame containing processed Jeopardy match data
        for the base case and both commercial scenarios, with a `scenario` column
        indicating the source scenario.
    """
    leverage_columns = ["top_1", "top_2", "top_4", "top_8", "bottom_3"]

    # Process base case
    clean_jeopardy_base_case_df = _get_latest_leverage_simulations(
        future_jeopardy_base_case_df, leverage_columns
    )
    clean_jeopardy_base_case_df = _process_jeopardy_dataframe(
        clean_jeopardy_base_case_df, False
    )
    clean_jeopardy_base_case_df["scenario"] = "status_quo"

    # Process commercial scenario 1
    clean_jeopardy_scenario_1_df = _get_latest_leverage_simulations(
        future_jeopardy_df_commercial_scenario_1, leverage_columns
    )
    clean_jeopardy_scenario_1_df = _process_jeopardy_dataframe(
        clean_jeopardy_scenario_1_df, False
    )
    clean_jeopardy_scenario_1_df["scenario"] = "uplift_to_average_squad"

    # Process commercial scenario 2
    clean_jeopardy_scenario_2_df = _get_latest_leverage_simulations(
        future_jeopardy_df_commercial_scenario_2, leverage_columns
    )
    clean_jeopardy_scenario_2_df = _process_jeopardy_dataframe(
        clean_jeopardy_scenario_2_df, False
    )
    clean_jeopardy_scenario_2_df["scenario"] = "uplift_to_top_teams"

    # Combine all processed DataFrames
    all_future_jeopardy_df = pd.concat(
        [
            clean_jeopardy_base_case_df,
            clean_jeopardy_scenario_1_df,
            clean_jeopardy_scenario_2_df,
        ],
        ignore_index=True,
    )

    # map names to be useable by model dataframe
    all_future_jeopardy_df["home_team"] = all_future_jeopardy_df[
        "home_team_name"
    ].replace(params["team_name_mapping_dict"])

    all_future_jeopardy_df["away_team"] = all_future_jeopardy_df[
        "away_team_name"
    ].replace(params["team_name_mapping_dict"])

    return all_future_jeopardy_df
