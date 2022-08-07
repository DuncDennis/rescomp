"""Streamlit elements to measure a time-series / a system. """

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px  # TODO: maybe move to plotly_plots

from streamlit_project.generalized_plotting import plotly_plots as plpl
from streamlit_project.app_fragments import utils
from streamlit_project.app_fragments import system_simulation
import rescomp.measures_new as meas


def st_statistical_measures(time_series_data: dict[str, np.ndarray]) -> None:
    """Streamlit element to calculate and plot statistical quantities of a time series.

    Args:
        time_series_data: The time series data.
    """
    utils.line()
    mode = st.selectbox("Statistical measure", ["std", "var", "mean", "median"])

    fig = plpl.statistical_barplot_multiple(time_series_data, mode=mode)
    st.plotly_chart(fig)
    utils.line()


@st.experimental_memo
def get_statistical_measure(time_series_dict: dict[str, np.ndarray],
                            mode: str = "std") -> pd.DataFrame:
    """Get a pandas DataFrame of a statistical quantity of a dict of time_series.
    # TODO: not really needed?
    Args:
        time_series_dict: The dict of time_series. The key is used as the legend label.
        mode: One of "std", "var", "mean", "median". # TODO more can be added.

    Returns:
        A Pandas DataFrame.
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

    return pd.DataFrame.from_dict(proc_data_dict)


@st.experimental_memo
def get_power_spectrum(time_series_dict: dict[str, np.ndarray], dt: float = 1.0,
                       per_or_freq: str = "period") -> pd.DataFrame:
    """Function to calculate the power spectrum of a time series dictionary.

    The pandas DataFrame returned has the following columns:
    - "period"/"frequency": The x axis to plot.
    - "label": One individual label for each time_series element (i.e. the key of the dict.).
    - "power 0" to "power {sys_dim-1}": The power of each time-series dimension.
    - "power_mean": The average over all dimensions.

    Args:
        time_series_dict: The input time series dictionary.
        dt: The time step.
        per_or_freq: Either "period" or "frequency". The x_axis of the spectrum.

    Returns:
        A Pandas DataFrame with all the power info.
    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape

    power_spectrum_dict = {per_or_freq: [], "label": [], "power_mean": []}
    power_spectrum_dict = power_spectrum_dict | {f"power {i}": [] for i in range(sys_dim)}
    for label, time_series in time_series_dict.items():
        if per_or_freq == "period":
            x, power_spectrum = meas.power_spectrum_componentwise(time_series, dt=dt, period=True)
        elif per_or_freq == "frequency":
            x, power_spectrum = meas.power_spectrum_componentwise(time_series, dt=dt, period=False)
        else:
            raise ValueError(f"This per_or_freq option is not accounted for.")

        power_spectrum_dict[per_or_freq] += x.tolist()
        power_spectrum_dict["label"] += [label, ] * x.size

        for i in range(sys_dim):
            power_spectrum_dict[f"power {i}"] += power_spectrum[:, i].tolist()
        power_spectrum_dict["power_mean"] += np.mean(power_spectrum, axis=1).tolist()

    return pd.DataFrame.from_dict(power_spectrum_dict)


def st_power_spectrum(time_series_dict: dict[str, np.ndarray], dt: float = 1.0) -> None:
    """Streamlit element to plot the power spectrum of a timeseries.

    Args:
        time_series_dict: The dictionary containing the time series data.
        dt: The time step of the timeseries.
    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape

    left, right = st.columns(2)
    with left:
        per_or_freq = st.selectbox("Period or Frequency", ["period", "frequency"])
    if per_or_freq == "period":
        log_x = True
    elif per_or_freq == "frequency":
        log_x = False
    else:
        raise ValueError(f"This per_or_freq option is not implemented.")

    df = get_power_spectrum(time_series_dict, dt=dt, per_or_freq=per_or_freq)

    opt = ["mean", "single dimension"]
    with right:
        opt_select = st.selectbox("Mean or single dimensions", opt)

    if opt_select == "single dimension":
        i_dim = utils.dimension_selection(sys_dim)
        label_to_plot = f"power {i_dim}"
    elif opt_select == "mean":
        label_to_plot = "power_mean"

    fig = plpl.plot_2d_line_or_scatter(to_plot_df=df, x_label=per_or_freq, y_label=label_to_plot,
                                 color="label", mode="line", title_i="Power Spectrum", log_x=log_x)
    st.plotly_chart(fig)


def st_largest_lyapunov_exponent(system_name: str, system_parameters: dict[str, Any]) -> None:
    """Streamlit element to calculate the largest lyapunov exponent.

    Set up the number inputs for steps, part_time_steps, steps_skip and deviation scale.
    Plot the convergence.

    # TODO maybe move to another python file like "system_measures"? Or just to measures?

    Args:
        system_name: The system name. Has to be in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
    """
    left, right = st.columns(2)
    with left:
        steps = int(st.number_input("steps", value=int(1e3)))
    with right:
        part_time_steps = int(st.number_input("time steps of each part", value=15))
    left, right = st.columns(2)
    with left:
        steps_skip = int(st.number_input("steps to skip", value=50, min_value=0))
    with right:
        deviation_scale = 10 ** (float(st.number_input("log (deviation_scale)", value=-10.0)))

    lle_conv = get_largest_lyapunov_exponent(system_name, system_parameters, steps=steps,
                                             part_time_steps=part_time_steps,
                                             deviation_scale=deviation_scale,
                                             steps_skip=steps_skip)
    largest_lle = np.round(lle_conv[-1], 5)

    figs = plpl.multiple_1d_time_series({"LLE convergence": lle_conv}, x_label="N",
                                        y_label="running avg of LLE", title=f"Largest Lyapunov "
                                                                            f"Exponent: "
                                                                            f"{largest_lle}")
    plpl.multiple_figs(figs)


@st.experimental_memo
def get_largest_lyapunov_exponent(system_name: str, system_parameters: dict[str, Any],
                                  steps: int = int(1e3), deviation_scale: float = 1e-10,
                                  part_time_steps: int = 15, steps_skip: int = 50) -> np.ndarray:
    """Measure the largest lyapunov exponent of a given system with specified parameters.

    Args:
        system_name: The system name. Has to be in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
        steps: The number of renormalization steps.
        deviation_scale: The initial deviation scale for nearby trajectories.
        part_time_steps: The nr of time steps between renormalizations.
        steps_skip: The nr of steps to skip before tracking the divergence.

    Returns:
        The convergence of the largest lyapunov with shape: (steps, ).
    """
    sim_instance = system_simulation.SYSTEM_DICT[system_name](**system_parameters)
    starting_point = sim_instance.default_starting_point

    if hasattr(sim_instance, "dt"):
        dt = sim_instance.dt
    else:
        dt = 1.0

    iterator_func = sim_instance.iterate
    lle_conv = meas.largest_lyapunov_exponent(iterator_func, starting_point=starting_point, dt=dt,
                                              steps=steps, part_time_steps=part_time_steps,
                                              deviation_scale=deviation_scale,
                                              steps_skip=steps_skip,
                                              return_convergence=True)
    return lle_conv


if __name__ == "__main__":
    pass
