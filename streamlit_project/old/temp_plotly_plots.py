""" A collection of utility plotting functions using plotly TEMPORARY -> to be refactored"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import rescomp.measures as measures


@st.experimental_memo
def plot_1d_time_series(time_series, i_dim, boundaries):
    names = ["train disc", "train sync", "train", "pred disc", "pred sync", "pred"]
    to_plot = time_series[:, i_dim]
    x_list = np.arange(time_series.shape[0])
    if boundaries is not None:
        right = 0
        fig = make_subplots()
        for i in range(6):
            left = right
            right = boundaries[i+1] + right

            fig.add_trace(go.Scatter(
                x=x_list[left: right],
                y=to_plot[left: right],
                name=names[i]
            )
            )
    return fig


def plot_valid_times_vs_pred_error(y_pred, y_true, error_thresh_min=0.1, error_thresh_max=1., steps=10, in_lyapunov_times=None):
    error_over_time = measures.error_over_time(y_pred, y_true, distance_measure="L2",
                                               normalization="root_of_avg_of_spacedist_squared")
    error_thresh_list = np.linspace(error_thresh_min, error_thresh_max, steps, endpoint=True)
    valid_times = np.zeros(steps)
    for i, thresh in enumerate(error_thresh_list):
        valid_time = measures.valid_time_index(error_over_time, thresh)
        valid_times[i] = valid_time

    if in_lyapunov_times is not None:
        dt = in_lyapunov_times["dt"]
        le = in_lyapunov_times["LE"]
        valid_times = dt*le*valid_times

    df = pd.DataFrame({"threshhold": error_thresh_list, "valid_times": valid_times})
    fig = px.line(df, x="threshhold", y="valid_times", title="valid_times vs error_threshhold")
    return fig
