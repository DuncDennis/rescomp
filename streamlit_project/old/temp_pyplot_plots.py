""" A collection of utility plotting functions using plotly"""
from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def plot_node_value_histogram(states, ax=None, title="", figsize=(8, 3), bins="auto"):
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        return_fig = False

    data = states.flatten()
    ax.hist(data, bins=bins)
    ax.set_title(title)
    if return_fig:
        return fig


@st.experimental_memo
def plot_node_value_histogram_multiple(states_data_dict, act_fct=None, figsize=(15, 7), bins="auto"):
    nr_of_hists = len(states_data_dict.keys())

    ncols = 2
    nrows = math.ceil(nr_of_hists/ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    for i, (title, states) in enumerate(states_data_dict.items()):
        i_col = i % ncols
        i_row = int(i/ncols)

        ax = axs[i_row, i_col]
        plot_node_value_histogram(states, ax=ax, title=title, bins=bins)
        if title == "act_fct_inp":
            if act_fct is not None:
                ax_twin = ax.twinx()
                x_lim_low, x_lim_high = ax.get_xlim()
                x_range = np.arange(x_lim_low, x_lim_high, 0.05)
                ax_twin.plot(x_range, act_fct(x_range), c="r")
    return fig
