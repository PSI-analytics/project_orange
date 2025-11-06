"""
This is a boilerplate pipeline 'model_preprocessing'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa

from .nodes import (
    create_model_df,
    create_viewership_model_df,
    scatter_plot_of_team_rating_vs_squad_rating,
    heatmap_of_jeopardy_over_time,
    create_simulation_team_dictionaries,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_model_df,
                inputs=[
                    "raw_attendance_data",
                    "player_elo_df",
                    "prospect_raw_jeopardy_df",
                    "params:model_preprocessing",
                ],
                outputs=[
                    "attendance_model_df",
                    "latest_team_ratings_df",
                ],
                name="create_model_df_node",
            ),
            node(
                func=scatter_plot_of_team_rating_vs_squad_rating,
                inputs="attendance_model_df",
                outputs="squad_rating_vs_team_rating_plot",
                name="scatter_plot_of_team_rating_vs_squad_rating_node",
            ),
            node(
                func=create_simulation_team_dictionaries,
                inputs="player_elo_df",
                outputs=[
                    "team_dictionary_commercial_scenario_1",
                    "team_dictionary_commercial_scenario_2",
                ],
                name="create_simulation_team_dictionaries_node",
            ),
            node(
                func=heatmap_of_jeopardy_over_time,
                inputs="prospect_raw_jeopardy_df",
                outputs=[
                    "jeopardy_heatmap_data",
                    "jeopardy_heatmap_relegation_plot",
                    "jeopardy_heatmap_title_plot",
                    "jeopardy_heatmap_playoff_plot",
                ],
                name="heatmap_of_jeopardy_over_time_node",
            ),
            node(
                func=create_viewership_model_df,
                inputs=[
                    "raw_viewership_data_sheet_1",
                    "raw_viewership_data_sheet_2",
                    "attendance_model_df",
                    "params:viewership_team_name_mapping_dict",
                ],
                outputs=[
                    "domestic_viewership_model_df",
                    "international_viewership_model_df",
                ],
                name="create_viewership_model_df_node",
            ),
        ]
    )
