from __future__ import annotations

import streamlit as st
import numpy as np
import inspect

from streamlit_project.generalized_plotting import plotly_plots as plpl
import rescomp.simulations_new as sims


SYSTEM_DICT = {
    "Lorenz63": sims.Lorenz63,
    "Roessler": sims.Roessler,
    "ComplexButterly": sims.ComplexButterly,
    "Chen": sims.Chen,
    "ChuaCircuit": sims.ChuaCircuit,
    "Thomas": sims.Thomas,
    "WindmiAttractor": sims.WindmiAttractor,
    "Rucklidge": sims.Rucklidge,
    "SimplestQuadraticChaotic": sims.SimplestQuadraticChaotic,
    "SimplestCubicChaotic": sims.SimplestCubicChaotic,
    "SimplestPiecewiseLinearChaotic": sims.SimplestPiecewiseLinearChaotic,
    "DoubleScroll": sims.DoubleScroll,
    "SimplestDrivenChaotic": sims.SimplestDrivenChaotic,
    "UedaOscillator": sims.UedaOscillator,
    "Henon": sims.Henon,
    "Logistic": sims.Logistic,
    "KuramotoSivashinsky": sims.KuramotoSivashinsky,
    "Lorenz96": sims.Lorenz96,
}


def st_select_system() -> tuple[str, dict]:
    """ Select the system to simulate and specify the parameters.

    Returns:
        tuple with system name and parameters dict

    """
    system_name = st.selectbox('System', SYSTEM_DICT.keys())

    sim_class = SYSTEM_DICT[system_name]

    signature_parameters = inspect.signature(sim_class.__init__).parameters

    fullargspec = inspect.getfullargspec(sim_class.__init__)
    system_parameters = {x: y for x, y in zip(fullargspec[0][1:], fullargspec[3])}

    with st.expander("Parameters: "):
        for key, val in system_parameters.items():
            type_str = signature_parameters[key].annotation
            if type_str == "float":
                system_parameters[key] = st.number_input(key, value=float(val), step=0.01, format="%f")
            elif type_str == "int":
                system_parameters[key] = st.number_input(key, value=int(val))
            else:
                raise TypeError("Other default keyword arguments than float and int are currently"
                                "not supported.")

    return system_name, system_parameters


def st_select_time_steps(default_time_steps: int = 10000) -> int:
    """
    Add number input for time steps.
    """
    return st.number_input('time_steps', value=default_time_steps, step=1)


def st_select_time_steps_split_up(default_t_train_disc: int = 1000,
                                  default_t_train_sync: int = 300,
                                  default_t_train: int = 2000,
                                  default_t_pred_disc: int = 1000,
                                  default_t_pred_sync: int = 300,
                                  default_t_pred: int = 5000,
                                  ) -> tuple[int, int, int, int, int, int]:
    """Add number inputs for train discard, train sync, train, pred discard, pred sync and pred t.

    """
    with st.expander("Time steps: "):
        t_train_disc = st.number_input('t_train_disc', value=default_t_train_disc, step=1)
        t_train_sync = st.number_input('t_train_sync', value=default_t_train_sync, step=1)
        t_train = st.number_input('t_train', value=default_t_train, step=1)
        t_pred_disc = st.number_input('t_pred_disc', value=default_t_pred_disc, step=1)
        t_pred_sync = st.number_input('t_pred_sync', value=default_t_pred_sync, step=1)
        t_pred = st.number_input('t_pred', value=default_t_pred, step=1)
        return t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred


@st.experimental_memo
def simulate_trajectory(system_name: str, system_parameters: dict, time_steps: int) -> np.ndarray:
    return SYSTEM_DICT[system_name](**system_parameters).simulate(time_steps=time_steps)


def default_simulation_plot(time_series):

    x_dim = time_series.shape[1]
    if x_dim == 1:

        figs = plpl.multiple_1d_time_series({"simulated timeseries": time_series, },
                                            x_label="time step",)
        plpl.multiple_figs(figs)

    elif x_dim == 2:
        fig = plpl.multiple_2d_time_series({"time_series": time_series}, mode="scatter")
        st.plotly_chart(fig)

    elif x_dim == 3:
        fig = plpl.multiple_3d_time_series({"time_series": time_series}, )
        st.plotly_chart(fig)

    elif x_dim > 3:
        figs = plpl.multiple_time_series_image({"time_series": time_series},
                                                x_label="time steps",
                                                y_label="dimensions"
                                               )
        plpl.multiple_figs(figs)
    else:
        raise ValueError("x_dim < 1 not supported.")


def main() -> None:
    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = st_select_system()
        time_steps = st_select_time_steps(default_time_steps=10000)

    time_series = simulate_trajectory(system_name, system_parameters, time_steps)

    default_simulation_plot(time_series)

    x_dim = time_series.shape[1]


if __name__ == '__main__':
    main()
