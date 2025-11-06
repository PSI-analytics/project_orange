"""
This is a boilerplate pipeline 'model_generation'
generated using Kedro 1.0.0
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from prospect.models.plotting import (
    create_model_predictions_scatter_plot,
)
from prospect.models.predictions import (
    create_model_ensemble_predictions_df,
    create_model_stacked_predictions_df,
)
from prospect.models.training import (
    evaluate_model_ensemble,
    train_model_ensemble,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("prospect")
logger.setLevel(logging.INFO)


def _convert_model_df_into_one_hot_encoded_model_df(
    model_df: pd.DataFrame,
    model_params: dict,
):
    """one hot encodes the model dataframe categorical columns and returns model dataframe plus new list of model column names

    Args:
        model_df: model dataframe
        model_params: dictionary of model parameters

    Returns:
        one_hot_encoded_model_df: one hot encoded model dataframe
        model_columns: list of new model columns

    """
    # one hot encode the categorical columns
    filtered_model_df = model_df[
        model_params["explainer_columns"]
        + [model_params["target_variable"]]
        + model_params["numerical_columns"]
    ]

    if model_params["categorical_columns"]:
        one_hot_encoded_model_df = pd.get_dummies(
            model_df[model_params["categorical_columns"]],
            columns=model_params["categorical_columns"],
            dtype=int,
        )

        one_hot_encoded_model_df = pd.concat(
            [filtered_model_df, one_hot_encoded_model_df], axis=1
        )
    else:
        one_hot_encoded_model_df = filtered_model_df.copy()

    model_columns = list(one_hot_encoded_model_df.columns)

    model_columns.remove(model_params["target_variable"])

    model_columns = [
        column
        for column in model_columns
        if column not in model_params["explainer_columns"]
    ]

    # Create a unique observation_id to be used throughout the pipeline
    # When the data is preprocessed, it needs to have a unique observation_id column for each row
    # This can be used to join back predictions at the end of the model training pipeline
    one_hot_encoded_model_df["observation_id"] = one_hot_encoded_model_df.index

    return one_hot_encoded_model_df, model_columns


def _standardise_model_df(
    model_df: pd.DataFrame,
    model_params: dict,
):
    """standardises the all numerical features in a given model dataframe

    Args:
        model_df: model dataframe
        model_params: dictionary of model params

    Returns:
        model_df: updated model dataframe which is now scaled
        scaler: scaler which can be used going forward when making model predictions

    """
    model_df_with_scaled_columns_df = model_df.copy(deep=True)

    model_df_scaled_df = model_df_with_scaled_columns_df[
        model_params["numerical_columns"]
    ]

    scaler = StandardScaler()

    model_df_scaled_df = pd.DataFrame(
        scaler.fit_transform(model_df_scaled_df),
        index=model_df_scaled_df.index,
        columns=model_df_scaled_df.columns,
    )

    model_df_with_scaled_columns_df[model_params["numerical_columns"]] = (
        model_df_scaled_df
    )

    return model_df_with_scaled_columns_df, scaler


def _train_and_evaluate_ensemble_model_with_predictions(
    model_df: pd.DataFrame,
    model_params: dict,
):
    """trains an ensemble model using the prospect package and returns evaluation metrics

    Args:
        model_df: model dataframe
        model_params: dictionary of relevant parameters for model training

    Returns:
        model_ensemble: ensemble prediction position model
        model_ensemble_model_scaler: scaler used for the prediction position model
        model_ensemble_coefficients_df: a concatenated dataframe of coefficients for the ensemble model
        model_ensemble_coefficients_summary_df: a summary dataframe of coefficients, with mean, std, max, min and 95% CI
        model_ensemble_predictions_df: the original dataframe, with train and test predictions for each observation
        model_ensemble_train_predictions_df: a dataframe of predictions for each model for each observation in the train sets
        model_ensemble_test_predictions_df: a dataframe of predictions for each model for each observation in the test sets
        model_ensemble_metrics_df: an aggregated dataframe showing the average train, test, bag and out-of-bag error metrics
        model_ensemble_train_metrics_df: a concatenated dataframe of train and test error metrics for each train set
        model_ensemble_bag_metrics_df: a concatenated dataframe of bag and out-of-bag error metrics for each train and bag set
        model_ensemble_coefficients_summary_plot: a bar chart of coefficients with their mean and 95% CI
        model_ensemble_train_predictions_scatter_plot: a scatter plot of actual vs. predicted values
        model_ensemble_aggregated_predictions_df: the original dataframe, with averaged predictions for each observation
        model_ensemble_individual_predictions_df: a dataframe with the predictions for each model for each observation

    """
    # standardize the model
    # currently because all the features are of a compariable scale ~100 standardisation makes no difference to the model performance
    if model_params["standardise_data"]:
        (
            standardise_model_df,
            model_ensemble_model_scaler,
        ) = _standardise_model_df(model_df, model_params)
    else:
        model_ensemble_model_scaler = pd.DataFrame()
        standardise_model_df = model_df.copy(deep=True)

    (
        one_hot_encoded_model_df,
        model_columns,
    ) = _convert_model_df_into_one_hot_encoded_model_df(
        standardise_model_df, model_params
    )

    # ensures pipeline can be switched between xgboost and linear model seamlessly
    if model_params["model_type"] == "xgboost_regression":
        model_ensemble = train_model_ensemble(
            one_hot_encoded_model_df,
            X_variables=model_columns,
            y_variable=model_params["target_variable"],
            model_name=model_params["model_name"],
            model_type=model_params["model_type"],
            random_state=model_params["random_state"],
            stacked_data=model_params["stacked_data"],
            stacked_columns=model_params["stacked_columns"],
            train_n_sets=model_params["train_n_sets"],
            bag_n_sets=model_params["bag_n_sets"],
            bag_n_splits=model_params["bag_n_splits"],
            bag_sample_with_replacement=model_params["bag_sample_with_replacement"],
            bag_split_size=model_params["bag_split_size"],
            xgboost_cv=model_params["xgboost_cv"],
            xgboost_learning_rate=model_params["xgboost_learning_rate"],
            xgboost_n_estimators=model_params["xgboost_n_estimators"],
            xgboost_max_depth=model_params["xgboost_max_depth"],
            xgboost_min_child_weight=model_params["xgboost_min_child_weight"],
            xgboost_regression_scoring=model_params["xgboost_regression_scoring"],
        )
    else:
        model_ensemble = train_model_ensemble(
            one_hot_encoded_model_df,
            X_variables=model_columns,
            y_variable=model_params["target_variable"],
            model_name=model_params["model_name"],
            model_type=model_params["model_type"],
            random_state=model_params["random_state"],
            stacked_data=model_params["stacked_data"],
            stacked_columns=model_params["stacked_columns"],
            train_n_sets=model_params["train_n_sets"],
            bag_n_sets=model_params["bag_n_sets"],
            bag_n_splits=model_params["bag_n_splits"],
            bag_sample_with_replacement=model_params["bag_sample_with_replacement"],
            bag_split_size=model_params["bag_split_size"],
            linear_regression_l1_ratio=1,
        )

    # evaluate model_ensemble
    (
        model_ensemble_coefficients_df,
        model_ensemble_coefficients_summary_df,
        model_ensemble_predictions_df,
        model_ensemble_train_predictions_df,
        model_ensemble_test_predictions_df,
        model_ensemble_metrics_df,
        model_ensemble_train_metrics_df,
        model_ensemble_bag_metrics_df,
    ) = evaluate_model_ensemble(
        one_hot_encoded_model_df,
        model_ensemble,
        model_error_metrics=model_params["model_error_metrics"],
        return_train_test_predictions_df=True,
        rename_predicted_columns_dict={},
    )

    if model_params["stacked_data"]:
        one_hot_encoded_model_df = one_hot_encoded_model_df.drop(columns="train_type")

    # The first dataframe contains the overall predictions, averaged across all ensemble models
    # The second dataframe contains the predictions for each model for each observation
    (
        model_ensemble_aggregated_predictions_df,
        model_ensemble_individual_predictions_df,
    ) = create_model_ensemble_predictions_df(
        one_hot_encoded_model_df,
        model_ensemble,
        return_individual_predictions_df=True,
        rename_predicted_columns_dict={},
    )

    model_ensemble_coefficients_summary_plot = _create_model_coefficients_summary_plot(
        model_ensemble_coefficients_summary_df,
        model_params["show_plots"],
    )

    # create scatter plot for model evaluation
    model_ensemble_train_predictions_scatter_plot = (
        create_model_predictions_scatter_plot(
            model_ensemble_predictions_df[
                model_ensemble_predictions_df["train_type"]
                == "test"  # to evaluate ensemble model on unseen data
            ],
            model_params["target_variable"],
            model_params["target_variable"] + "_predicted",
            model_params["show_plots"],
        )
    )

    return (
        model_ensemble,
        model_ensemble_model_scaler,
        model_ensemble_coefficients_df,
        model_ensemble_coefficients_summary_df,
        model_ensemble_predictions_df,
        model_ensemble_train_predictions_df,
        model_ensemble_test_predictions_df,
        model_ensemble_metrics_df,
        model_ensemble_train_metrics_df,
        model_ensemble_bag_metrics_df,
        model_ensemble_coefficients_summary_plot,
        model_ensemble_train_predictions_scatter_plot,
        model_ensemble_aggregated_predictions_df,
        model_ensemble_individual_predictions_df,
    )


def _plot_rolling_averages(
    df: pd.DataFrame,
    variable_name_1: str,
    variable_name_2: str,
    date_column_name: str,
    rolling_average_window: int,
    season: str,
    block: bool = False,
    label_interval: int = 50,
) -> plt.Figure:
    """Plot rolling averages of two variables over time with periodic date labels.

    Creates a line plot showing the rolling averages of two columns against an index axis.
    The x-axis displays dates in 'Mon-YY' format (e.g., 'Jan-25', 'Feb-25') at regular
    intervals (every N points) to avoid overcrowding.

    Args:
        df: Input DataFrame containing the data to plot.
        variable_name_1: Name of the first column to compute rolling average for.
            Plotted as a solid green line.
        variable_name_2: Name of the second column to compute rolling average for.
            Plotted as a dotted grey line.
        date_column_name: Name of the date column to use for the x-axis labels.
        rolling_average_window: Size of the rolling window for computing averages.
        season: Season string to filter the data (e.g., "2024/2025").
        block: If True, block execution and run GUI main loop until figure is closed.
            If False, display figure and return immediately. Defaults to False.
            See matplotlib.pyplot.show() documentation for details.
        label_interval: Interval for displaying date labels on x-axis. Labels will
            appear every N points. Defaults to 25.

    Returns:
        Matplotlib Figure object containing the plot.

    Examples:
        >>> fig = _plot_rolling_averages(
        ...     df=match_data,
        ...     variable_name_1="attendance_linear_regression_predicted",
        ...     variable_name_2="attendance",
        ...     date_column_name="match_date",
        ...     rolling_average_window=50,
        ...     season="2024/2025",
        ...     block=False,
        ...     label_interval=25,
        ... )

    Notes:
        - Date labels appear every N points (configurable via label_interval)
        - The function filters data to the specified season before plotting
        - Rolling averages are computed before season filtering
        - Points are plotted continuously regardless of actual date gaps
        - NaN values in rolling averages (at the start) are handled automatically
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Ensure the date column is in datetime format
    df[date_column_name] = pd.to_datetime(df[date_column_name])

    # Sort by date to ensure correct plotting order
    df = df.sort_values(by=date_column_name).reset_index(drop=True)

    # Compute rolling averages
    df[f"{variable_name_1}_rolling"] = (
        df[variable_name_1].rolling(window=rolling_average_window).mean()
    )
    df[f"{variable_name_2}_rolling"] = (
        df[variable_name_2].rolling(window=rolling_average_window).mean()
    )

    # Filter by season
    df = df[df["season"] == season].reset_index(drop=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the rolling averages using index on x-axis
    ax.plot(
        df.index,
        df[f"{variable_name_1}_rolling"],
        label=f"{variable_name_1} ({rolling_average_window}-day rolling avg)",
        color="#22D081",
        linewidth=2,
    )
    ax.plot(
        df.index,
        df[f"{variable_name_2}_rolling"],
        label=f"{variable_name_2} ({rolling_average_window}-day rolling avg)",
        color="grey",
        linestyle="dotted",
        linewidth=2,
    )

    # Set custom x-axis ticks and labels (every label_interval points)
    tick_positions = list(range(0, len(df), label_interval))
    tick_labels = [
        df[date_column_name].iloc[i].strftime("%b-%y") for i in tick_positions
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Formatting the plot
    ax.set_title(
        f"Rolling Averages of {variable_name_1} and {variable_name_2} Over Time",
        fontsize=14,
        pad=20,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Rolling Average", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show(block=block)

    return fig


def _create_model_coefficients_summary_plot(
    df: pd.DataFrame,
    block=None,
):
    """
    Takes a summary dataframe of coefficients, with mean, std, max, min and 95% CI of coefficients and other
    parameters, and creates a bar chart of coefficients with their mean and 95% CI.

    Parameters
    ----------
    df : pd.DataFrame
        A summary dataframe of coefficients, with mean, std, max, min and 95% CI.
    block: bool, optional
        block paramater for matplotlib.pyplot.show.(). Whether to wait for all figures to be closed before returning. If True block and run the GUI main loop until all figure windows are closed.
        If False ensure that all figure windows are displayed and return immediately. In this case, you are responsible
        for ensuring that the event loop is running to have responsive figures. For more information see the `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html>`_.

    Returns
    -------
    plt
        A bar chart of coefficients with their mean and 95% CI.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> from prospect.models.plotting import _create_model_coefficients_summary_plot
        >>> data = {
        ...     "statistic": [
        ...         "mean",
        ...         "std",
        ...         "max",
        ...         "min",
        ...         "lower_95_CI",
        ...         "upper_95_CI",
        ...     ],
        ...     "feature_1": [0.2, 0.1, 0.5, -0.1, 0.1, 0.3],
        ...     "feature_2": [0.3, 0.2, 0.6, 0.0, 0.1, 0.5],
        ...     "feature_3": [-0.1, 0.1, 0.2, -0.3, -0.2, 0.0],
        ... }
        >>> df = pd.DataFrame(data)
        >>> plot = _create_model_coefficients_summary_plot(df)  # doctest: +SKIP
    """

    # Create a clean dataframe, with statistic as the index (which will be the column names when transposed)
    model_coefficients_summary_df = df.copy()
    model_coefficients_summary_df.index = model_coefficients_summary_df["statistic"]

    # Create a list of columns to drop from the dataframe for the plot, and loop through them
    drop_column_names = [
        "model_name",
        "model_type",
        "y_name",
        "intercept",
        "l1_ratio",
        "alpha",
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_child_weight",
    ]
    for column_name in drop_column_names:
        try:
            model_coefficients_summary_df.drop(columns=[column_name], inplace=True)
        except Exception:
            pass

    # Transpose, reset index and sort by the mean value of the coefficients
    model_coefficients_summary_df = model_coefficients_summary_df.drop(
        columns=["statistic"]
    ).T.reset_index()
    model_coefficients_summary_df.sort_values("mean", ascending=False, inplace=True)

    # Define prospect_bar_color_map
    prospect_bar_color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mycmap", ["#22D081", "#A2E4BF"]
    )

    # Set the color_map for the bar chart (NB can be set to tab20c)
    color_map = prospect_bar_color_map(range(len(model_coefficients_summary_df)))
    # color_map = plt.cm.tab20c(range(len(model_coefficients_summary_df)))

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.bar(
        model_coefficients_summary_df["index"],
        model_coefficients_summary_df["mean"],
        align="center",
        alpha=1,
        ecolor="black",
        capsize=4,
        color=color_map,
    )

    # Annotate the bar chart
    ax.set_ylabel("Coefficient", size=12)
    ax.set_xlabel("Variable", size=12)
    ax.set_title(
        "Coefficient for each variable, with 95% confidence intervals",
        size=14,
        weight="bold",
    )

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Show the plot
    plt.tight_layout()
    plt.show(block=block)

    return fig


def create_stacked_model(
    model_df: pd.DataFrame,
    linear_regression_model_params: dict,
    xgboost_regression_model_params: dict,
):
    """creates a predictive attendance model and returns associated evaluation metrics
    method for model is to predict stadium capacity and then convert this into attendance to ensure
    that the model never predicts over a plausible stadium capacity

    Args:
        model_df: model dataframe
        linear_regression_model_params: dictionary of relevant parameters for first layer model training

    Returns:
        predicted_attendance_model_ensemble: ensemble prediction attendance model
        predicted_attendance_model_scaler: scaler used for the prediction attendance model
        predicted_attendance_model_coefficients_df: a concatenated dataframe of coefficients for the ensemble model
        predicted_attendance_model_coefficients_summary_df: a summary dataframe of coefficients, with mean, std, max, min and 95% CI
        predicted_attendance_model_predictions_df: the original dataframe, with train and test predictions for each observation
        predicted_attendance_model_train_predictions_df: a dataframe of predictions for each model for each observation in the train sets
        predicted_attendance_model_test_predictions_df: a dataframe of predictions for each model for each observation in the test sets
        predicted_attendance_model_metrics_df: an aggregated dataframe showing the average train, test, bag and out-of-bag error metrics
        predicted_attendance_model_train_metrics_df: a concatenated dataframe of train and test error metrics for each train set
        predicted_attendance_model_bag_metrics_df: a concatenated dataframe of bag and out-of-bag error metrics for each train and bag set
        predicted_attendance_model_coefficients_summary_plot: a bar chart of coefficients with their mean and 95% CI
        predicted_attendance_model_stadium_capacity_scatter_plot: a scatter plot of actual vs. predicted values in stadium capacity space
        predicted_attendance_linear_model_attendance_scatter_plot: a scatter plot of actual vs. predicted values in attendance space
        model_df_with_all_model_predictions: the original dataframe, with ensemble predictions for each observation

    """
    model_df = model_df[
        ~model_df["season"].isin(linear_regression_model_params["holdout_season"])
    ].reset_index(drop=True)

    (
        predicted_attendance_model_ensemble,
        predicted_attendance_model_scaler,
        predicted_attendance_model_coefficients_df,
        predicted_attendance_model_coefficients_summary_df,
        predicted_attendance_model_predictions_df,
        predicted_attendance_model_train_predictions_df,
        predicted_attendance_model_test_predictions_df,
        predicted_attendance_model_metrics_df,
        predicted_attendance_model_train_metrics_df,
        predicted_attendance_model_bag_metrics_df,
        predicted_attendance_model_coefficients_summary_plot,
        predicted_attendance_model_stadium_capacity_scatter_plot,
        linear_regression_predicted_earnings_model_ensemble_predictions_df,
        linear_regression_predicted_earnings_model_individual_predictions_df,
    ) = _train_and_evaluate_ensemble_model_with_predictions(
        model_df, linear_regression_model_params
    )

    predicted_attendance_model_stacked_predictions_df = (
        create_model_stacked_predictions_df(
            predicted_attendance_model_predictions_df,
            [predicted_attendance_model_train_predictions_df],
            [predicted_attendance_model_test_predictions_df],
            [[linear_regression_model_params["target_variable"] + "_predicted"]],
        )
    )

    predicted_attendance_model_stacked_predictions_df = (
        predicted_attendance_model_stacked_predictions_df.rename(
            columns={
                linear_regression_model_params["target_variable"]
                + "_predicted": linear_regression_model_params["target_variable"]
                + "_"
                + linear_regression_model_params["model_type"]
                + "_predicted"
            }
        )
    )

    (
        xgboost_predicted_attendance_model_ensemble,
        xgboost_predicted_attendance_model_scaler,
        xgboost_predicted_attendance_model_coefficients_df,
        xgboost_predicted_attendance_model_coefficients_summary_df,
        xgboost_predicted_attendance_model_predictions_df,
        xgboost_predicted_attendance_model_train_predictions_df,
        xgboost_predicted_attendance_model_test_predictions_df,
        xgboost_predicted_attendance_model_metrics_df,
        xgboost_predicted_attendance_model_train_metrics_df,
        xgboost_predicted_attendance_model_bag_metrics_df,
        xgboost_predicted_attendance_model_coefficients_summary_plot,
        xgboost_predicted_attendance_model_stadium_capacity_scatter_plot,
        xgboost_predicted_earnings_model_ensemble_predictions_df,
        xgboost_predicted_earnings_model_individual_predictions_df,
    ) = _train_and_evaluate_ensemble_model_with_predictions(
        predicted_attendance_model_stacked_predictions_df,
        xgboost_regression_model_params,
    )

    linear_regression_predicted_earnings_model_ensemble_predictions_df = (
        linear_regression_predicted_earnings_model_ensemble_predictions_df.rename(
            columns={
                linear_regression_model_params["target_variable"]
                + "_predicted": linear_regression_model_params["target_variable"]
                + "_"
                + linear_regression_model_params["model_type"]
                + "_predicted"
            }
        )
    )

    # Create xgboost_regression_score_diff_model_ensemble_predictions_df
    # The first dataframe contains the overall predictions, averaged across all ensemble models
    # The second dataframe contains the predictions for each model for each observation
    (
        all_regression_predicted_earnings_model_ensemble_predictions_df,
        _,
    ) = create_model_ensemble_predictions_df(
        linear_regression_predicted_earnings_model_ensemble_predictions_df,
        xgboost_predicted_attendance_model_ensemble,
        return_individual_predictions_df=False,
        rename_predicted_columns_dict={},
    )

    # save model predictions on to non one-hot encoded dataframe
    model_df_with_all_model_predictions = model_df.copy(deep=True)

    model_df_with_all_model_predictions[
        linear_regression_model_params["target_variable"]
        + "_"
        + linear_regression_model_params["model_type"]
        + "_predicted"
    ] = linear_regression_predicted_earnings_model_ensemble_predictions_df[
        linear_regression_model_params["target_variable"]
        + "_"
        + linear_regression_model_params["model_type"]
        + "_predicted"
    ]

    model_df_with_all_model_predictions[
        xgboost_regression_model_params["target_variable"]
        + "_"
        + xgboost_regression_model_params["model_type"]
        + "_predicted"
    ] = all_regression_predicted_earnings_model_ensemble_predictions_df[
        xgboost_regression_model_params["target_variable"] + "_predicted"
    ]

    # predict attendance for linear model
    # as using a general pipeline create logic for plotting attendance even though we are looking to predict venue_capacity
    # this could not apply for viewership
    if linear_regression_model_params["target_variable"] == "venue_percentage_capacity":
        model_df_with_all_model_predictions[
            "attendance_" + linear_regression_model_params["model_type"] + "_predicted"
        ] = (
            model_df_with_all_model_predictions[
                linear_regression_model_params["target_variable"]
                + "_"
                + linear_regression_model_params["model_type"]
                + "_predicted"
            ]
            * model_df_with_all_model_predictions["venue_capacity"]
        )
        # Create xgboost_regression_score_diff_model_train_predictions_scatter_plot
        predicted_attendance_linear_model_attendance_scatter_plot = (
            create_model_predictions_scatter_plot(
                model_df_with_all_model_predictions,
                "attendance",
                "attendance_"
                + linear_regression_model_params["model_type"]
                + "_predicted",
                linear_regression_model_params["show_plots"],
            )
        )
        linear_regression_all_league_rolling_average_plot = _plot_rolling_averages(
            model_df_with_all_model_predictions,
            variable_name_1="attendance_"
            + linear_regression_model_params["model_type"]
            + "_predicted",
            variable_name_2="attendance",
            date_column_name="match_date",
            rolling_average_window=50,
            season="2024/2025",
            block=linear_regression_model_params["show_plots"],
        )
    else:
        predicted_attendance_linear_model_attendance_scatter_plot = plt.figure()
        linear_regression_all_league_rolling_average_plot = _plot_rolling_averages(
            model_df_with_all_model_predictions,
            variable_name_1=linear_regression_model_params["target_variable"]
            + "_"
            + linear_regression_model_params["model_type"]
            + "_predicted",
            variable_name_2=linear_regression_model_params["target_variable"],
            date_column_name="match_date",
            rolling_average_window=50,
            season="2024/2025",
            block=linear_regression_model_params["show_plots"],
        )

    # predict attendance for xgboost model
    if (
        xgboost_regression_model_params["target_variable"]
        == "venue_percentage_capacity"
    ):
        model_df_with_all_model_predictions[
            "attendance_" + xgboost_regression_model_params["model_type"] + "_predicted"
        ] = (
            model_df_with_all_model_predictions[
                xgboost_regression_model_params["target_variable"]
                + "_"
                + xgboost_regression_model_params["model_type"]
                + "_predicted"
            ]
            * model_df_with_all_model_predictions["venue_capacity"]
        )
        predicted_attendance_xgboost_model_attendance_scatter_plot = (
            create_model_predictions_scatter_plot(
                model_df_with_all_model_predictions,
                "attendance",
                "attendance_"
                + xgboost_regression_model_params["model_type"]
                + "_predicted",
                xgboost_regression_model_params["show_plots"],
            )
        )
        xgboost_regression_all_league_rolling_average_plot = _plot_rolling_averages(
            model_df_with_all_model_predictions,
            variable_name_1="attendance_"
            + xgboost_regression_model_params["model_type"]
            + "_predicted",
            variable_name_2="attendance",
            date_column_name="match_date",
            rolling_average_window=50,
            season="2024/2025",
            block=xgboost_regression_model_params["show_plots"],
        )
    else:
        predicted_attendance_xgboost_model_attendance_scatter_plot = plt.figure()
        xgboost_regression_all_league_rolling_average_plot = _plot_rolling_averages(
            model_df_with_all_model_predictions,
            variable_name_1=xgboost_regression_model_params["target_variable"]
            + "_"
            + xgboost_regression_model_params["model_type"]
            + "_predicted",
            variable_name_2=xgboost_regression_model_params["target_variable"],
            date_column_name="match_date",
            rolling_average_window=50,
            season="2024/2025",
            block=xgboost_regression_model_params["show_plots"],
        )

    return (
        predicted_attendance_model_ensemble,
        predicted_attendance_model_scaler,
        predicted_attendance_model_coefficients_df,
        predicted_attendance_model_coefficients_summary_df,
        predicted_attendance_model_predictions_df,
        predicted_attendance_model_train_predictions_df,
        predicted_attendance_model_test_predictions_df,
        predicted_attendance_model_metrics_df,
        predicted_attendance_model_train_metrics_df,
        predicted_attendance_model_bag_metrics_df,
        predicted_attendance_model_coefficients_summary_plot,
        predicted_attendance_model_stadium_capacity_scatter_plot,
        predicted_attendance_linear_model_attendance_scatter_plot,
        xgboost_predicted_attendance_model_ensemble,
        xgboost_predicted_attendance_model_scaler,
        xgboost_predicted_attendance_model_coefficients_df,
        xgboost_predicted_attendance_model_coefficients_summary_df,
        xgboost_predicted_attendance_model_predictions_df,
        xgboost_predicted_attendance_model_train_predictions_df,
        xgboost_predicted_attendance_model_test_predictions_df,
        xgboost_predicted_attendance_model_metrics_df,
        xgboost_predicted_attendance_model_train_metrics_df,
        xgboost_predicted_attendance_model_bag_metrics_df,
        xgboost_predicted_attendance_model_coefficients_summary_plot,
        xgboost_predicted_attendance_model_stadium_capacity_scatter_plot,
        predicted_attendance_xgboost_model_attendance_scatter_plot,
        model_df_with_all_model_predictions,
        linear_regression_all_league_rolling_average_plot,
        xgboost_regression_all_league_rolling_average_plot,
    )
