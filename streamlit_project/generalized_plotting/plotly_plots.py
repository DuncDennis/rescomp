""" A collection of utility plotting functions using plotly"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def matrix_as_barchart(data_matrix: np.ndarray, x_axis: str = "x_dim", y_axis: str = "y_dim",
                       value_name: str = "value", title: str = "",
                       fig_size: tuple[int, int] = (650, 500),
                       log_y: bool = False, abs: bool = True, barmode: str = "relative"
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
        abs: If true the absolute value of data_matrix entries is used.
        barmode: If "relative" the values corresponding to the different y_axis_indices are plotted
                in one bar chart and are stacked on top of each other. If "grouped" they are
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

    if abs:
        abs_value_name = f"absolute of {value_name}"
        df[abs_value_name] = np.abs(df[value_name])
        value_col_to_plot = abs_value_name
    else:
        value_col_to_plot = value_name

    if barmode in ["relative", "grouped"]:
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

