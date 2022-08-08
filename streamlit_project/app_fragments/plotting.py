"""Python file that includes Streamit elements used for plotting the timeseries."""

from __future__ import annotations

import numpy as np

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
