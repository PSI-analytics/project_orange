"""
This is a boilerplate pipeline 'model_generation'
generated using Kedro 1.0.0
"""

from project_orange import settings
from kedro.pipeline import Node, Pipeline, node, pipeline  # noqa

from .nodes import (
    create_stacked_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    train_stacked_model_pipeline = Pipeline(
        [
            node(
                func=create_stacked_model,
                inputs=[
                    "model_df",
                    "params:linear_regression_model_params",
                    "params:xgboost_regression_model_params",
                ],
                outputs=[
                    "linear_regression_model",
                    "linear_regression_model_scaler",
                    "linear_regression_model_coefficients_df",
                    "linear_regression_model_coefficients_summary_df",
                    "linear_regression_model_predictions_df",
                    "linear_regression_model_train_predictions_df",
                    "linear_regression_model_test_predictions_df",
                    "linear_regression_model_metrics_df",
                    "linear_regression_model_train_metrics_df",
                    "linear_regression_model_bag_metrics_df",
                    "linear_regression_xgboost_model_feature_importance",
                    "linear_regression_scatter_plot_of_model_vs_actual_train",
                    "linear_regression_scatter_plot_of_attendance_vs_predicted_attendance",
                    "xgboost_regression_model",
                    "xgboost_regression_model_scaler",
                    "xgboost_regression_model_coefficients_df",
                    "xgboost_regression_model_coefficients_summary_df",
                    "xgboost_regression_model_predictions_df",
                    "xgboost_regression_model_train_predictions_df",
                    "xgboost_regression_model_test_predictions_df",
                    "xgboost_regression_model_metrics_df",
                    "xgboost_regression_model_train_metrics_df",
                    "xgboost_regression_model_bag_metrics_df",
                    "xgboost_regression_xgboost_model_feature_importance",
                    "xgboost_regression_scatter_plot_of_model_vs_actual_train",
                    "xgboost_regression_scatter_plot_of_attendance_vs_predicted_attendance",
                    "all_regression_model_ensemble_predictions_df",
                    "linear_regression_rolling_average_plot",
                    "xgboost_regression_rolling_average_plot",
                ],
                name="create_stacked_model_node",
            ),
        ]
    )

    pipes = []
    for namespace, variants in settings.STACKED_MODEL_PIPELINES_MAPPING.items():
        for variant in variants:
            pipes.append(
                pipeline(
                    train_stacked_model_pipeline,
                    namespace=f"{namespace}.{variant}",
                    tags=[variant, namespace, "model_generation"],
                )
            )

    return sum(pipes)
