from datetime import datetime

import pandas as pd
import pandera as pa
from pandera import extensions
from pandera.typing import Series
from typing import Optional


class attendance_model_schema(pa.DataFrameModel):
    """
    Data schema for the predicted stadium capacity model using pandera.

    This schema validates the structure and types of the DataFrame used for
    predicting stadium capacities. Each field corresponds to a specific
    feature of the model, ensuring that the values conform to the expected
    types and constraints.

    """

    season: Series[str] = pa.Field(coerce=True)
    match_date: Series[pd.Timestamp] = pa.Field(
        coerce=True,
    )
    month: Series[str] = pa.Field(
        coerce=True,
        isin=[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    )
    day_of_week: Series[str] = pa.Field(
        coerce=True,
    )
    week_of_month: Series[int] = pa.Field(
        ge=1,
        le=5,
        coerce=True,
    )
    match_kick_off_time: Series[str] = pa.Field(coerce=True)
    time_slot: Series[str] = pa.Field(coerce=True)
    home_team: Series[str] = pa.Field(coerce=True)
    away_team: Series[str] = pa.Field(coerce=True)
    venue_name: Series[str] = pa.Field(coerce=True)
    venue_capacity: Series[int] = pa.Field(coerce=True)
    derby: Series[int] = pa.Field(coerce=True)
    prospect_home_team_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_away_team_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_home_squad_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_away_squad_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_combined_team_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_combined_squad_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_team_rating_abs_difference: Series[float] = pa.Field(ge=0, coerce=True)
    home_team_win_prob: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    match_jeopardy_title: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    match_jeopardy_relegation: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    match_jeopardy_play_offs: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    total_match_jeopardy: Optional[Series[float]] = pa.Field(ge=0, le=2, coerce=True)
    attendance: Series[int] = pa.Field(coerce=True)
    venue_percentage_capacity: Series[float] = pa.Field(coerce=True)

    # will filter the given dataframe to only contain the columns specified above
    class Config:
        strict = "filter"


class viewership_model_schema(pa.DataFrameModel):
    """
    Data schema for the predicted viewership model using pandera.

    This schema validates the structure and types of the DataFrame used for
    predicting viewership. Each field corresponds to a specific
    feature of the model, ensuring that the values conform to the expected
    types and constraints.

    """

    season: Series[str] = pa.Field(coerce=True)
    match_date: Series[pd.Timestamp] = pa.Field(
        coerce=True,
    )
    month: Series[str] = pa.Field(
        coerce=True,
        isin=[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    )
    day_of_week: Series[str] = pa.Field(
        coerce=True,
    )
    week_of_month: Series[int] = pa.Field(
        ge=1,
        le=5,
        coerce=True,
    )
    match_kick_off_time: Series[str] = pa.Field(coerce=True)
    time_slot: Series[str] = pa.Field(coerce=True)
    home_team: Series[str] = pa.Field(coerce=True)
    away_team: Series[str] = pa.Field(coerce=True)
    venue_name: Series[str] = pa.Field(coerce=True)
    venue_capacity: Series[int] = pa.Field(coerce=True)
    derby: Series[int] = pa.Field(coerce=True)
    prospect_home_team_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_away_team_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_home_squad_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_away_squad_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_combined_team_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_combined_squad_rating: Series[float] = pa.Field(ge=0, coerce=True)
    prospect_team_rating_abs_difference: Series[float] = pa.Field(ge=0, coerce=True)
    home_team_win_prob: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    match_jeopardy_title: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    match_jeopardy_relegation: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    match_jeopardy_play_offs: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    total_match_jeopardy: Optional[Series[float]] = pa.Field(ge=0, le=2, coerce=True)
    viewership: Series[int] = pa.Field(ge=0, coerce=True)

    # will filter the given dataframe to only contain the columns specified above
    class Config:
        strict = "filter"
