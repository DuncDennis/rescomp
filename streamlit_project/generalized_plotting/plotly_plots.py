""" A collection of utility plotting functions using plotly"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


DEFAULT_TWO_D_FIGSIZE = (650, 350)
DEFAULT_THREE_D_FIGSIZE = (650, 500)


def multiple_figs(figs: list[go.Figure, ...]) -> None:
    """Utility function to plot multiple figs in streamlit.
    Args:
        figs: List of plotly figures.

    """
    for fig in figs:
        st.plotly_chart(fig)


@st.experimental_memo
def matrix_as_barchart(data_matrix: np.ndarray, x_axis: str = "x_dim", y_axis: str = "y_dim",
                       value_name: str = "value", title: str = "",
                       fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                       log_y: bool = False, abs_bool: bool = True, barmode: str = "relative"
                       ) -> go.Figure:
    """Plot the absolut values of a matrix as a relative/grouped/subplotted barchart.


    Args:
        data_matrix: 2 dimensional numpy array to visualize.
        x_axis: Name of the x-axis index of the data_matrix. Will be displayed as the x-axis of the
                bar-plot.
        y_axis: Name of the y-axis index of the data_matrix. Will be displayed above the colorsbar.
        value_name: Name of the values within the data_matrix.
        title: Title of the plot.
        fig_size: The size of the figure in (width, height).
        log_y: If true the y axis of the plot will be displayed logarithmically.
        abs_bool: If true the absolute value of data_matrix entries is used.
        barmode: If "relative" the values corresponding to the different y_axis_indices are plotted
                in one bar chart and are stacked on top of each other. If "group" they are
                plotted next to each other. If "subplot" there is a new subplot for every y_axis
                index.

    Returns:
        plotly figure.
    """

    x_dim, y_dim = data_matrix.shape

    data_dict = {x_axis: [], y_axis: [], value_name: []}
    for i_x in range(x_dim):
        for i_y in range(y_dim):
            value = data_matrix[i_x, i_y]
            data_dict[x_axis].append(i_x)
            data_dict[value_name].append(value)
            data_dict[y_axis].append(i_y)

    df = pd.DataFrame.from_dict(data_dict)

    if abs_bool:
        abs_value_name = f"absolute of {value_name}"
        df[abs_value_name] = np.abs(df[value_name])
        value_col_to_plot = abs_value_name
    else:
        value_col_to_plot = value_name

    if barmode in ["relative", "group"]:
        fig = px.bar(df, x=x_axis, y=value_col_to_plot, color=y_axis,
                     title=title, width=fig_size[0],
                     height=fig_size[1], barmode=barmode)

    elif barmode == "subplot":
        subplot_titles = [f"{title} - {y_axis}: {i_y}" for i_y in range(y_dim)]
        fig = make_subplots(rows=y_dim, cols=1, subplot_titles=subplot_titles)
        for i_y in range(y_dim):
            sub_df = df[df[y_axis] == i_y]
            sub_fig = px.bar(sub_df, x=x_axis, y=value_col_to_plot)

            fig.add_trace(sub_fig["data"][0], row=i_y+1, col=1)
        fig.update_layout(height=fig_size[1]*y_dim, width=fig_size[0])
        fig.update_yaxes(title=value_col_to_plot)
        fig.update_xaxes(title=x_axis)

    else:
        raise ValueError(f"Value of keyword argument barmode = {barmode} is not supported.")

    if log_y:
        fig.update_yaxes(type="log", exponentformat="E")

    return fig


def _plot_2d_line_or_scatter(to_plot_df: pd.DataFrame, x_label: str, y_label: str,
                             mode: str, color: str, title_i: str,
                             line_size: float | None = 1,
                             scatter_size: float | None = 1,
                             fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                             log_x: bool = False, log_y: bool = False
                             ) -> go.Figure:
    """General utility plotting function for 2d plots.
    TODO: Better docstring.
    """
    if mode == "line":
        fig = px.line(to_plot_df, x=x_label, y=y_label, color=color,
                      title=title_i)
        if line_size is not None:
            fig.update_traces(line={"width": line_size})
    elif mode == "scatter":
        fig = px.scatter(to_plot_df, x=x_label, y=y_label, color=color,
                         title=title_i)
        if scatter_size is not None:
            fig.update_traces(marker={'size': scatter_size})
    else:
        raise Exception(f"mode = {mode} not accounted for.")  # TODO: proper error

    fig.update_layout(height=fig_size[1], width=fig_size[0])
    if log_y:
        fig.update_layout(
            yaxis={
                'exponentformat': 'E'}
        )
    if log_x:
        fig.update_layout(
            xaxis={
                'exponentformat': 'E'}
        )

    return fig


@st.experimental_memo
def multiple_1d_time_series(time_series_dict: dict[str, np.ndarray], mode: str = "line",
                            line_size: float | None = 1, scatter_size: float | None = 1,
                            title: str | None = None,
                            fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                            x_scale: float | None = None, x_label: str = "steps",
                            y_label: str = "value", dimensions: tuple[int, ...] | None = None,
                            subplot_dimensions_bool: bool = True
                            ) -> list[go.Figure]:
    """ Plot multiple 1d time_series as a line or scatter plot.
    # TODO: add possibility for vertical lines seperators.
    # TODO: add logy and logx setting.
    # TODO: maybe generalize to not only plot "time_series" but general 2d data.

    Args:
        time_series_dict: Dict of the form {"timeseries_name_1": time_series_1, ...}.
        mode: "line" or "scatter".
        line_size: If mode = "line" size of lines.
        scatter_size: If mode = "scatter", size of markers.
        title: Title of figure
        fig_size: The size of the figure in (width, height).
        x_scale: Scale the x axis. (Probably time axis).
        x_label: x_label for xaxis.
        y_label: y_label for yaxis.
        dimensions: If the timeseries is multidimensional specify the dimensions to plot. If None
                    All dimensions are plotted beneath each other.
        subplot_dimensions_bool: If true: Make a new fig for each dimension. Else: make a new fig
                                for each entry in the time_series_dict. (Flip the plot).

    Returns: plotly figure.

    """
    figs = []
    shape = list(time_series_dict.values())[0].shape
    if len(shape) == 1:
        x_steps, y_dim = shape[0], 1
    else:
        x_steps, y_dim = shape
    print(x_steps, y_dim)

    if x_scale is None:
        x_array = np.arange(x_steps)
    else:
        x_array = np.arange(x_steps) * x_scale

    if dimensions is None:
        dimensions = tuple(np.arange(y_dim))

    to_plot_dict = {x_label: [], y_label: [], "label": [], "dimension": []}
    for dim in dimensions:
        for label, time_series in time_series_dict.items():
            if len(shape) == 1:
                time_series = time_series[:, np.newaxis]
            to_plot_dict[x_label].extend(x_array)
            to_plot_dict[y_label].extend(time_series[:, dim])
            to_plot_dict["label"].extend([label, ] * time_series.shape[0])
        to_plot_dict["dimension"].extend([dim, ] * time_series.shape[0] * len(time_series_dict))

    df = pd.DataFrame.from_dict(to_plot_dict)

    if subplot_dimensions_bool:
        for dim in dimensions:
            df_selection = df[df["dimension"] == dim]
            if len(dimensions) > 1:
                if title is not None:
                    title_i = f"{title}: dimension: {dim}"
                else:
                    title_i = f"Dimension: {dim}"
            else:
                title_i = title

            color = "label"
            fig = _plot_2d_line_or_scatter(df_selection, x_label, y_label, mode,
                                           color, title_i, line_size=line_size,
                                           scatter_size=scatter_size,
                                           fig_size=fig_size)
            figs.append(fig)

    else:
        labels_list = list(time_series_dict.keys())
        for label in labels_list:
            df_selection = df[df["label"] == label]

            if title is not None:
                title_i = f"{title}: {label}"
            else:
                title_i = f"{label}"

            color = "dimension"

            fig = _plot_2d_line_or_scatter(df_selection, x_label, y_label, mode,
                                           color, title_i, line_size=line_size,
                                           scatter_size=scatter_size,
                                           fig_size=fig_size)
            figs.append(fig)

    return figs


@st.experimental_memo
def multiple_2d_time_series(time_series_dict: dict[str, np.ndarray], mode: str = "line",
                            line_size: float | None = 1, scatter_size: float | None = 1,
                            title: str | None = None,
                            fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                            x_label: str = "x", y_label: str = "y",
                            ) -> go.Figure:
    """ Plot multiple 2d time_series as a line or scatter plot.

    Args:
        time_series_dict: Dict of the form {"timeseries_name_1": time_series_1, ...}.
        mode: "line" or "scatter".
        line_size: If mode = "line" size of lines.
        scatter_size: If mode = "scatter", size of markers.
        title: Title of figure
        fig_size: The size of the figure in (width, height).
        x_label: x_label for xaxis.
        y_label: y_label for yaxis.

    Returns: plotly figure.

    """

    to_plot_dict = {x_label: [], y_label: [], "label": []}
    for label, time_series in time_series_dict.items():
        to_plot_dict[x_label].extend(time_series[:, 0])
        to_plot_dict[y_label].extend(time_series[:, 1])
        to_plot_dict["label"].extend([label, ] * time_series.shape[0])

    if mode == "line":
        fig = px.line(to_plot_dict, x=x_label, y=y_label, color="label", title=title)
        if line_size is not None:
            fig.update_traces(line={"width": line_size})
    elif mode == "scatter":
        fig = px.scatter(to_plot_dict, x=x_label, y=y_label, color="label", title=title)
        if scatter_size is not None:
            fig.update_traces(marker={'size': scatter_size})
    else:
        raise Exception(f"mode = {mode} not accounted for.")  # TODO: proper error

    fig.update_layout(height=fig_size[1], width=fig_size[0])
    return fig


@st.experimental_memo
def multiple_3d_time_series(time_series_dict: dict[str, np.ndarray], mode: str = "line",
                            line_size: float | None = 1, scatter_size: float | None = 1,
                            title: str | None = None,
                            fig_size: tuple[int, int] = DEFAULT_THREE_D_FIGSIZE,
                            x_label: str = "x", y_label: str = "y", z_label: str = "z"
                            ) -> go.Figure:
    """ Plot multiple 3d time_series as a line or scatter plot.

    Args:
        time_series_dict: Dict of the form {"timeseries_name_1": time_series_1, ...}.
        mode: "line" or "scatter".
        line_size: If mode = "line" size of lines.
        scatter_size: If mode = "scatter", size of markers.
        title: Title of figure
        fig_size: The size of the figure in (width, height).
        x_label: x_label for xaxis.
        y_label: y_label for yaxis.
        z_label: z_label for zaxis.

    Returns: plotly figure.

    """
    to_plot_dict = {x_label: [], y_label: [], z_label: [], "label": []}
    for label, time_series in time_series_dict.items():
        to_plot_dict[x_label].extend(time_series[:, 0])
        to_plot_dict[y_label].extend(time_series[:, 1])
        to_plot_dict[z_label].extend(time_series[:, 2])
        to_plot_dict["label"].extend([label, ] * time_series.shape[0])

    if mode == "line":
        fig = px.line_3d(to_plot_dict, x=x_label, y=y_label, z=z_label, color="label", title=title)
        if line_size is not None:
            fig.update_traces(line={"width": line_size})
    elif mode == "scatter":
        fig = px.scatter_3d(to_plot_dict, x=x_label, y=y_label, z=z_label, color="label",
                            title=title)
        if scatter_size is not None:
            fig.update_traces(marker={'size': scatter_size})
    else:
        raise Exception(f"mode = {mode} not accounted for.")  # TODO: proper error

    fig.update_layout(height=fig_size[1], width=fig_size[0])
    return fig


@st.experimental_memo
def multiple_time_series_image(time_series_dict: dict[str, np.ndarray],
                               fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE,
                               x_label: str = "x",
                               y_label: str = "y",
                               x_scale: float | None = None,
                               ) -> list[go.Figure]:
    # TODO: add docstring
    # TODO: add possibility for vertical lines seperators
    figs = []
    labels = {"x": x_label, "y": y_label}

    if x_scale is None:
        x_array = None
    else:
        time_steps = list(time_series_dict.values())[0].shape[0]
        x_array = np.arange(time_steps) * x_scale

    for i, (key, val) in enumerate(time_series_dict.items()):
        figs.append(
            px.imshow(val.T, aspect="auto", title=key, width=fig_size[0], height=fig_size[1],
                      labels=labels, x=x_array)
        )
    return figs


@st.experimental_memo
def statistical_barplot_multiple(time_series_dict: dict[str, np.ndarray],
                                 mode: str = "std",
                                 x_label: str = "system dimension",
                                 title: str | None = None,
                                 fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE) -> go.Figure:
    """Plot a statistical quantity of a dict of time_series as a grouped barplot.

    Args:
        time_series_dict: The dict of time_series. The key is used as the legend label.
        mode: One of "std", "var", "mean", "median". # TODO more can be added.
        x_label: The name of the x_axis.
        title: The title of the plot.
        fig_size: The figure size.

    Returns:
        The plotly figure.
    """

    time_steps, sys_dim = list(time_series_dict.values())[0].shape

    proc_data_dict = {"x_axis": [], "label": [], mode: []}
    for label, data in time_series_dict.items():
        if mode == "std":
            stat_quant = np.std(data, axis=0)
        elif mode == "mean":
            stat_quant = np.mean(data, axis=0)
        elif mode == "median":
            stat_quant = np.median(data, axis=0)
        elif mode == "var":
            stat_quant = np.var(data, axis=0)
        else:
            raise ValueError(f"Mode {mode} is not implemented.")

        proc_data_dict["x_axis"] += np.arange(sys_dim).tolist()
        proc_data_dict["label"] += [label, ] * sys_dim
        proc_data_dict[mode] += stat_quant.tolist()

    df = pd.DataFrame.from_dict(proc_data_dict)

    fig = px.bar(df, x="x_axis", y=mode, color="label", barmode="group",
                 title=title)
    fig.update_layout(height=fig_size[1], width=fig_size[0])
    fig.update_xaxes(title=x_label)

    return fig


def power_spectrum_multiple(time_series_dict: dict[str, np.ndarray],
                            ) -> go.Figure:
    time_steps, sys_dim = list(time_series_dict.values())[0].shape

    to_plot_dict = {"x": [], "y": [], "label": []}

    for label, data in time_series_dict.items():
        to_plot_dict["label"] += [label, ] * sys_dim
        x = None
# def multiple_1d_data_xy(data_dict: dict[str, tuple[np.ndarray, np.ndarray]],
#                         log_x: bool = False, log_y: bool = False,
#                         title: str | None = None, x_label: str = "x",
#                         y_label = "y",
#                         fig_size: tuple[int, int] = DEFAULT_THREE_D_FIGSIZE,
#                         )

# @st.experimental_memo
# def mean_and_std_barplot(time_series: np.ndarray,
#                          fig_size: tuple[int, int] = DEFAULT_TWO_D_FIGSIZE) -> list[go.Figure]:
#     """Plot the mean and standard deviation of the time series as a bar plot.
#     TODO: enhance so that one can compare multiple timeseries. Maybe remove
#     Args:
#         time_series: The input time series of shape (time_steps, sys_dim).
#         fig_size: The size of the figure in (width, height).
#
#
#     Returns:
#         List of both plotly figures for mean and std.
#     """
#
#     x_axis_title = "system dimension"
#
#     mean = np.mean(time_series, axis=0)[:, np.newaxis]
#     std = np.std(time_series, axis=0)[:, np.newaxis]
#
#     mean_all = np.round(np.mean(mean), 6)
#     std_all = np.round(np.std(time_series), 6)
#
#     mean_fig = px.bar(mean, width=fig_size[0],
#                      height=fig_size[1], title=f"Mean of time series: {mean_all}")
#     mean_fig.update_yaxes(title="mean")
#     mean_fig.update_xaxes(title=x_axis_title)
#     mean_fig.update_layout(showlegend=False)
#
#     std_fig = px.bar(std, width=fig_size[0],
#                       height=fig_size[1], title=f"Std of time series: {std_all}")
#     std_fig.update_yaxes(title="std")
#     std_fig.update_xaxes(title=x_axis_title)
#     std_fig.update_layout(showlegend=False)
#
#     return [mean_fig, std_fig]
