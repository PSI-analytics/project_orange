"""
This is a boilerplate pipeline 'model_predictions'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa

from .nodes import (
    run_commercial_scenarios,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=run_commercial_scenarios,
                inputs=[
                    "attendance_model_df",
                    "domestic_viewership_model_df",
                    "international_viewership_model_df",
                    "stacked_model.attendance.linear_regression_model",
                    "stacked_model.attendance.linear_regression_model_scaler",
                    "params:stacked_model.attendance.linear_regression_model_params",
                    "stacked_model.attendance.xgboost_regression_model",
                    "stacked_model.attendance.xgboost_regression_model_scaler",
                    "params:stacked_model.attendance.xgboost_regression_model_params",
                    "stacked_model.domestic_viewership.linear_regression_model",
                    "stacked_model.domestic_viewership.linear_regression_model_scaler",
                    "params:stacked_model.domestic_viewership.linear_regression_model_params",
                    "stacked_model.domestic_viewership.xgboost_regression_model",
                    "stacked_model.domestic_viewership.xgboost_regression_model_scaler",
                    "params:stacked_model.domestic_viewership.xgboost_regression_model_params",
                    "stacked_model.international_viewership.linear_regression_model",
                    "stacked_model.international_viewership.linear_regression_model_scaler",
                    "params:stacked_model.international_viewership.linear_regression_model_params",
                    "stacked_model.international_viewership.xgboost_regression_model",
                    "stacked_model.international_viewership.xgboost_regression_model_scaler",
                    "params:stacked_model.international_viewership.xgboost_regression_model_params",
                    "params:commercial_simulations_params",
                ],
                outputs="model_predictions_for_commercial_scenario",
                name="run_commercial_scenarios_node",
            ),
        ]
    )
