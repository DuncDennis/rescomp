"""Python file that includes Streamlit elements used for plotting esn quantites."""

from __future__ import annotations

from typing import Callable

import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from streamlit_project.generalized_plotting import plotly_plots as plpl
from streamlit_project.app_fragments import utils
from streamlit_project.app_fragments import measures as meas_app
import streamlit_project.latex_formulas.esn_formulas as esn_latex


def st_plot_w_out_as_barchart(w_out: np.ndarray, key: str | None = None) -> None:
    """Streamlit element to plot w_out as a barchart.

    TODO: add bargap as a option in matrix_as_barchart?

    Args:
        w_out: The w_out matrix of shape (output dimension, r_gen dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    log_y = st.checkbox("log y", key=f"{key}__st_plot_w_out_as_barchart__logy")
    fig = plpl.matrix_as_barchart(w_out.T, x_axis="r_gen index", y_axis="out dim",
                                  value_name="w_out", log_y=log_y)
    fig.update_layout(bargap=0.0)
    st.plotly_chart(fig)


def st_plot_architecture(x_dim: int, r_dim: int, r_gen_dim: int, y_dim: int) -> None:
    """Streamlit element to plot dimensions of the layers in the esn.

    Args:
        x_dim: The input dimension of the esn.
        r_dim: The reservoir dimension.
        r_gen_dim: The generalized reservoir dimension.
        y_dim: The output dimension.

    """
    utils.st_line()
    cols = st.columns(7)
    cols[0].markdown("**Input:**")

    cols[1].latex(r"\rightarrow")

    cols[2].markdown("**Reservoir states:**")

    cols[3].latex(r"\rightarrow")

    cols[4].markdown("**Generalized res. states:**")

    cols[5].latex(r"\rightarrow")
    cols[6].markdown("**Output:**")

    cols = st.columns(7)

    cols[0].markdown(f"**{x_dim}**")

    cols[2].markdown(f"**{r_dim}**")

    cols[4].markdown(f"**{r_gen_dim}**")

    cols[6].markdown(f"**{y_dim}**")

    utils.st_line()


def st_reservoir_states_histogram(res_train_dict: dict[str, np.ndarray],
                                  res_pred_dict: dict[str, np.ndarray],
                                  act_fct: Callable[[np.ndarray], np.ndarray] | None,
                                  key: str | None = None) -> None:
    """Streamlit element to show histograms of reservoir state quantities.

    TODO: Bad practice that res_train_dict has one item more than actually needed?
    TODO: Also include bias and r_to_r_gen?

    Show value histograms of:
    - Res. input: W_in * INPUT
    - Res. internal update: Network * PREVIOUS_RES_STATE
    - Act. fct. argument: W_in * INPUT + Network * PREVIOUS_RES_STATE + BIAS.
    - Res. states: The reservoir states.

    All values are first flattened over all nodes and then the histogram is created.

    Args:
        res_train_dict: A dictionary containing "r_input", "r_internal", "r_act_fct_inp", "r"
                        corresponding to the reservoir state quantities during training.
        res_pred_dict: A dictionary containing "r_input", "r_internal", "r_act_fct_inp", "r"
                       corresponding to the reservoir state quantities during prediction.
        act_fct: The activation function used in the esn.
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    utils.st_line()
    st.latex(esn_latex.w_in_and_network_update_equation_with_explanation)
    utils.st_line()

    cols = st.columns(3)
    with cols[0]:
        train_or_predict = st.selectbox("Train or predict", ["train", "predict"],
                                        key=f"{key}__st_reservoir_states_histogram__top")
    with cols[1]:
        bins = int(st.number_input("Bins", min_value=2, value=50,
                                   key=f"{key}__st_reservoir_states_histogram__bins"))
    with cols[2]:
        share_x = st.checkbox("Share x", key=f"{key}__st_reservoir_states_histogram__sharex")
        share_y = st.checkbox("Share y", key=f"{key}__st_reservoir_states_histogram__sharey")
        if share_x:
            share_x = "all"
        if share_y:
            share_y = "all"

    if train_or_predict == "train":
        res_state_dict = res_train_dict
    elif train_or_predict == "predict":
        res_state_dict = res_pred_dict
    else:
        raise ValueError("This train or predict option is not accounted for.")

    res_state_dict_flattened = {key: val.flatten()[:, np.newaxis] for key, val in
                                res_state_dict.items() if key != "r_gen"}
    df = meas_app.get_histograms(res_state_dict_flattened, dim_selection=[0], bins=bins)

    fig = make_subplots(rows=2, cols=2, shared_xaxes=share_x, shared_yaxes=share_y,
                        subplot_titles=["Res. input", "Res. internal update", "Act. fct. argument",
                                        "Res. states"],
                        specs=[[{}, {}],
                               [{"secondary_y": True}, {}]],
                        horizontal_spacing=0.1,
                        vertical_spacing=0.2)

    df_sub = df[df["label"] == "r_input"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=1, col=1
    )

    df_sub = df[df["label"] == "r_internal"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=1, col=2
    )

    df_sub = df[df["label"] == "r_act_fct_inp"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=2, col=1
    )
    bins = df_sub["bins"]
    fig.add_trace(
        go.Scatter(x=bins, y=act_fct(bins), showlegend=True, name="activation function",
                   mode="lines"),
        secondary_y=True,
        row=2, col=1
    )

    df_sub = df[df["label"] == "r"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=2, col=2
    )

    fig.update_layout(bargap=0.0)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="left")
        )
    fig.update_layout(width=750, height=500)

    st.plotly_chart(fig)
