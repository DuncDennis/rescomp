"""Python file that includes Streamit elements used for plotting the timeseries."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_project.generalized_plotting import plotly_plots as plpl
from streamlit_project.app_fragments import utils


def st_plot_dim_selection(time_series_dict: dict[str, np.ndarray]) -> None:
    """Streamlit element to plot a selection of dimensions of timeseries as 1d timeseries.

    Args:
        time_series_dict: The dictionary containing the timeseries.

    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape
    dim_selection = utils.st_dimension_selection_multiple(dimension=sys_dim)
    figs = plpl.multiple_1d_time_series(time_series_dict, mode="line",
                                        line_size=1,
                                        dimensions=tuple(dim_selection))
    plpl.multiple_figs(figs)


def st_default_simulation_plot(time_series):
    """Streamlit element to plot a time series independent of shape.

    If 1d, plot value vs. time.
    If 2d, plot value_1 vs value_2 as a scatter plot.
    If 3d, plot value_1 vs value_2 vs value_3 as a line plot.
    If d>3, plot as a heatmap: values vs time.

    Args:
        time_series: The timeseries of shape (time_steps, sys_dim)
    """

    x_dim = time_series.shape[1]
    if x_dim == 1:

        figs = plpl.multiple_1d_time_series({"simulated timeseries": time_series, },
                                            x_label="time step",)
        plpl.multiple_figs(figs)

    elif x_dim == 2:
        fig = plpl.multiple_2d_time_series({"simulated timeseries": time_series}, mode="scatter")
        st.plotly_chart(fig)

    elif x_dim == 3:
        fig = plpl.multiple_3d_time_series({"simulated timeseries": time_series}, )
        st.plotly_chart(fig)

    elif x_dim > 3:
        figs = plpl.multiple_time_series_image({"simulated timeseries": time_series},
                                               x_label="time steps",
                                               y_label="dimensions"
                                               )
        plpl.multiple_figs(figs)
    else:
        raise ValueError("x_dim < 1 not supported.")
