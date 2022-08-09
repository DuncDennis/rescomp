"""Python file that includes Streamlit elements used for plotting esn quantites."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_project.generalized_plotting import plotly_plots as plpl
# from streamlit_project.app_fragments import utils


def st_plot_w_out_as_barchart(w_out: np.ndarray, key: str | None = None) -> None:
    """Streamlit element to plot w_out as a barchart.

    Args:
        w_out: The w_out matrix of shape (output dimension, r_gen dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    log_y = st.checkbox("log y", key=f"{key}__st_plot_w_out_as_barchart__logy")
    fig = plpl.matrix_as_barchart(w_out.T, x_axis="r_gen index", y_axis="out dim",
                                  value_name="w_out", log_y=log_y)
    st.plotly_chart(fig)
