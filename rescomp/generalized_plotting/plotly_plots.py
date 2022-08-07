import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


def as_barchart(data_matrix: np.ndarray, x_axis="x_dim", y_axis="y_dim", value_name="value",
                    title="", fig_size=(650, 500), log_y=False, abs=True, stacked=False):
    """plot the absolut values of a matrix as a stacked barchart

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

    if stacked:
        fig = px.bar(df, x=x_axis, y=value_col_to_plot, color=y_axis,
                     title=title, width=fig_size[0],
                     height=fig_size[1])

    else:
        subplot_titles = [f"{title} - {y_axis}: {i_y}" for i_y in range(y_dim)]
        fig = make_subplots(rows=y_dim, cols=1, subplot_titles=subplot_titles)
        for i_y in range(y_dim):
            sub_df = df[df[y_axis] == i_y]
            sub_fig = px.bar(sub_df, x=x_axis, y=value_col_to_plot)

            fig.add_trace(sub_fig["data"][0], row=i_y+1, col=1)
        fig.update_layout(height=fig_size[1]*y_dim, width=fig_size[0])
        fig.update_yaxes(title=value_col_to_plot)
        fig.update_xaxes(title=x_axis)
    if log_y:
        fig.update_yaxes(type="log", exponentformat="E")

    return fig
