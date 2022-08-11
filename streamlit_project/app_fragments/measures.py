"""Streamlit elements to measure a time-series / a system. """

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from streamlit_project.generalized_plotting import plotly_plots as plpl
from streamlit_project.app_fragments import utils
from streamlit_project.app_fragments import system_simulation
import rescomp.measures_new as meas

STATE_PREFIX = "measures"  # The prefix for the session state variables.


@st.experimental_memo
def get_histograms(time_series_dict: dict[str, np.ndarray], dim_selection: list[int],
                   bins: int = 50) -> pd.DataFrame:
    """Calculate a histogram of a time_series_dict, return a pandas Dataframe.

    Args:
        time_series_dict: The dictionary containing the time_series data.
        dim_selection: A list of integers specifying the dimensions to consider.
        bins: The number of bins.

    Returns:
        A pandas DataFrame with the columns: "label" (key in time_series_dict), "bins" (the bin
        middle points), "histogram" (the histogram values), "dimension" (the dimension of the
        system they refer to).
    """

    data_dict = {"label": [], "bins": [], "histogram": [], "dimension": []}

    for i_dim in dim_selection:
        for key, val in time_series_dict.items():
            hist, binedges = np.histogram(val[:, i_dim], bins=bins, density=True)
            binmiddle = np.array([(binedges[i] + binedges[i + 1]) / 2 for i in range(bins)])
            nr_entries = hist.size
            data_dict["bins"] += binmiddle.tolist()
            data_dict["histogram"] += hist.tolist()
            data_dict["label"] += [key, ] * nr_entries
            data_dict["dimension"] += [i_dim, ] * nr_entries

    return pd.DataFrame.from_dict(data_dict)


def st_histograms(time_series_dict: dict[str, np.ndarray],
                  key: str | None = None) -> None:
    """Streamlit element to plot a histogram of the time_series.

    There is a bins element and a "dimension selector".

    Args:
        time_series_dict: The dictionary containing the time series data.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape
    left, right = st.columns(2)
    with left:
        dim_selection = utils.st_dimension_selection_multiple(sys_dim, key=f"{key}__st_histogram")
    with right:
        bins = int(st.number_input("Bins", min_value=2, value=50, key=f"{key}__st_histogram__bins"))
    data_df = get_histograms(time_series_dict, dim_selection, bins=bins)
    for i_dim in dim_selection:
        sub_df = data_df[data_df["dimension"] == i_dim]
        fig = plpl.barplot(sub_df, x="bins", y="histogram", color="label",
                           title=f"Histogram: Dim = {i_dim}",
                           x_label="value", y_label="frequency", barmode="overlay")
        st.plotly_chart(fig)


@st.experimental_memo
def get_extrema_maps(time_series_dict: dict[str, np.ndarray], dim_selection: list[int],
                     mode: str = "maximum"
                     ) -> list[dict[str, np.ndarray]]:
    """ Get the extrema data as a list of dictionaries: One dict for each dimension.

    For each dimension selected in dim_selection, create a dictionary with the same keys as
    time_series_dict, but values of the shape (nr of extrema, 2) for consecutive extrema.

    Args:
        time_series_dict: The dictionary containing the time series.
        dim_selection: A list of ints specifying the dimensions you want to plot.
        mode: Either "minima" or "maxima".

    Returns:
        A list of dictionaries. Each dictionary can create one plot.
    """
    sub_dicts = []
    for i_dim in dim_selection:
        sub_dicts.append({key: meas.extrema_map(val, mode=mode, i_dim=i_dim) for key, val in
                          time_series_dict.items()})
    return sub_dicts


def st_extrema_map(time_series_dict: dict[str, np.ndarray], key: str | None = None) -> None:
    """A streamlit element to plot the extrema map of a time_series dict.

    Args:
        time_series_dict: The dictionary containing the time series.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape
    left, right = st.columns(2)
    with left:
        dim_selection = utils.st_dimension_selection_multiple(sys_dim,
                                                              key=f"{key}__st_extrema_map")
    with right:
        mode = st.selectbox("Min or max", ["minima", "maxima"])
    sub_dicts = get_extrema_maps(time_series_dict, dim_selection=dim_selection, mode=mode)
    for dim, sub_dict in zip(dim_selection, sub_dicts):
        fig = plpl.multiple_2d_time_series(sub_dict, mode="scatter", title=f"Dim = {dim}",
                                           x_label="extreme value (i)",
                                           y_label="extreme value (i+1)", scatter_size=2)
        st.plotly_chart(fig)


def st_statistical_measures(time_series_dict: dict[str, np.ndarray], key: str | None = None
                            ) -> None:
    """Streamlit element to calculate and plot statistical quantities of a time series.

    Args:
        time_series_dict: The time series data.
        key: Provide a unique key if this streamlit element is used multiple times.
    """
    mode = st.selectbox("Statistical measure", ["std", "var", "mean", "median"],
                        key=f"{key}__st_statistical_measures")

    df = get_statistical_measure(time_series_dict, mode=mode)
    fig = plpl.barplot(df, x="x_axis", y=mode, color="label",
                       x_label="system dimension")

    st.plotly_chart(fig)


@st.experimental_memo
def get_statistical_measure(time_series_dict: dict[str, np.ndarray],
                            mode: str = "std") -> pd.DataFrame:
    """Get a pandas DataFrame of a statistical quantity of a dict of time_series.
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


def st_power_spectrum(time_series_dict: dict[str, np.ndarray], dt: float = 1.0,
                      key: str | None = None) -> None:
    """Streamlit element to plot the power spectrum of a timeseries.

    Args:
        time_series_dict: The dictionary containing the time series data.
        dt: The time step of the timeseries.
        key: Provide a unique key if this streamlit element is used multiple times.
    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape

    left, right = st.columns(2)
    with left:
        per_or_freq = st.selectbox("Period or Frequency", ["period", "frequency"],
                                   key=f"{key}__st_power_spectrum__per_or_freq")
    if per_or_freq == "period":
        log_x = True
    elif per_or_freq == "frequency":
        log_x = False
    else:
        raise ValueError(f"This per_or_freq option is not implemented.")

    df = get_power_spectrum(time_series_dict, dt=dt, per_or_freq=per_or_freq)

    opt = ["mean", "single dimension"]
    with right:
        opt_select = st.selectbox("Mean or single dimensions", opt,
                                  key=f"{key}__st_power_spectrum__opt_select")

    if opt_select == "single dimension":
        dim_selection = utils.st_dimension_selection_multiple(sys_dim,
                                                              key=f"{key}__st_power_spectrum")
        labels_to_plot = [f"power {i_dim}" for i_dim in dim_selection]
    elif opt_select == "mean":
        dim_selection = ["all", ]
        labels_to_plot = ["power_mean", ]
    else:
        raise ValueError("Selected option is not accounted for. ")

    for i_dim, label_to_plot in zip(dim_selection, labels_to_plot):
        fig = plpl.plot_2d_line_or_scatter(to_plot_df=df, x_label=per_or_freq,
                                           y_label=label_to_plot,
                                           color="label", mode="line",
                                           title_i=f"Power Spectrum, dim = {i_dim}", log_x=log_x)
        st.plotly_chart(fig)


def st_largest_lyapunov_exponent(system_name: str, system_parameters: dict[str, Any],
                                 key: str | None = None) -> None:
    """Streamlit element to calculate the largest lyapunov exponent.

    Set up the number inputs for steps, part_time_steps, steps_skip and deviation scale.
    Plot the convergence.

    # TODO maybe move to another python file like "system_measures"? Or just to measures?

    Args:
        system_name: The system name. Has to be in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
        key: Provide a unique key if this streamlit element is used multiple times.
    """
    left, right = st.columns(2)
    with left:
        steps = int(st.number_input("steps", value=int(1e3),
                                    key=f"{key}__st_largest_lyapunov_exponent__steps"))
    with right:
        part_time_steps = int(st.number_input("time steps of each part", value=15,
                                              key=f"{key}__st_largest_lyapunov_exponent__part"))
    left, right = st.columns(2)
    with left:
        steps_skip = int(st.number_input("steps to skip", value=50, min_value=0,
                                         key=f"{key}__st_largest_lyapunov_exponent__skip"))
    with right:
        deviation_scale = 10 ** (float(st.number_input("log (deviation_scale)", value=-10.0,
                                                       key=f"{key}__st_largest_lyapunov_exponent__eps")))

    lle_conv = get_largest_lyapunov_exponent(system_name, system_parameters, steps=steps,
                                             part_time_steps=part_time_steps,
                                             deviation_scale=deviation_scale,
                                             steps_skip=steps_skip)
    largest_lle = np.round(lle_conv[-1], 5)

    utils.st_add_to_state(prefix=STATE_PREFIX, name="LLE", value=largest_lle)

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


@st.experimental_memo
def get_largest_lyapunov_from_data(time_series_dict: dict[str, np.ndarray],
                                   time_steps: int = 100,
                                   dt: float = 1.0,
                                   neighbours_to_check: int = 50,
                                   min_index_difference: int = 50,
                                   distance_upper_bound: float = np.inf
                                   ) -> pd.DataFrame:
    """Calculate the largest lyapunov based only data.

    Args:
        time_series_dict: The dictionary containing the time-series data.
        time_steps: The nr of time steps to use for the divergence.
        dt: The time step.
        neighbours_to_check: Number of next neighbours to check so that there is atleast one that
                             fullfills the min_index_difference condition.
        min_index_difference: The minimal difference between neighbour indices in order to count as
                              a valid neighbour.
        distance_upper_bound: If the distance is too big it is also not a valid neighbour.

    Returns:
        A pandas DataFrame with the columns: "log_div/dt", "label" and "steps".
    """
    out_dict = {"label": [], "log_div/dt": [], "steps": [], "linear fit": [], "lle": []}
    for label, time_series in time_series_dict.items():
        log_dist, _ = meas.largest_lyapunov_from_data(time_series,
                                                      time_steps=time_steps,
                                                      dt=dt,
                                                      neighbours_to_check=neighbours_to_check,
                                                      min_index_difference=min_index_difference,
                                                      distance_upper_bound=distance_upper_bound)
        nr_steps = log_dist.size
        steps = np.arange(nr_steps)

        coefficients = np.polyfit(steps, log_dist, 1)
        poly1d_fn = np.poly1d(coefficients)
        out_dict["linear fit"] += poly1d_fn(steps).tolist()

        lyap = np.polyfit(steps, log_dist, 1)[0]
        out_dict["lle"] += [np.round(lyap, 4), ] * nr_steps
        out_dict["log_div/dt"] += log_dist.tolist()
        # new_label = f"{label}: {np.round(lyap, 4)}"
        new_label = f"{label}"
        out_dict["label"] += [new_label, ] * nr_steps
        out_dict["steps"] += steps.tolist()

    return pd.DataFrame.from_dict(out_dict)


def st_largest_lyapunov_from_data(time_series_dict: dict[str, np.ndarray],
                                  dt: float = 1.0,
                                  key: str | None = None) -> None:
    """A streamlit element to calculate and plot the largest lyapunov exponent based on data.

    Args:
        time_series_dict: The dictionary containing the time series data.
        dt: The time step dt.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    col1, col2, col3 = st.columns(3)
    with col1:
        time_steps = int(st.number_input("Time steps", value=100, min_value=2,
                                         key=f"{key}__st_largest_lyapunov_from_data__ts"))
    with col2:
        neighbours_to_check = int(st.number_input("Neighbours to check", value=50, min_value=1,
                                                  key=f"{key}__st_largest_lyapunov_from_data__nc"))
    with col3:
        min_index_difference = int(st.number_input("Min index difference", value=50, min_value=1,
                                                   key=f"{key}__st_largest_lyapunov_from_data__mid"))
    df_to_plot = get_largest_lyapunov_from_data(time_series_dict, time_steps=time_steps,
                                                neighbours_to_check=neighbours_to_check,
                                                min_index_difference=min_index_difference,
                                                dt=dt)

    col_pal = px.colors.qualitative.Plotly
    col_pal_iterator = itertools.cycle(col_pal)
    fig = go.Figure()
    for label in df_to_plot["label"].unique():
        df_sub = df_to_plot[df_to_plot["label"] == label]
        color = next(col_pal_iterator)
        lle = df_sub["lle"].unique()[0]

        fig.add_trace(
            go.Scatter(x=df_sub["steps"], y=df_sub["log_div/dt"], name=label, showlegend=True,
                       legendgroup=label,
                       line=dict(color=color)
                       )
        )

        fig.add_trace(
            go.Scatter(x=df_sub["steps"], y=df_sub["linear fit"],
                       name=f"linear fit: sloap/lyapunov: {lle}",
                       legendgroup=label,
                       showlegend=True,
                       line=dict(color=color, dash='dash', width=1),
                       marker=dict()
                       )
        )

    fig.update_xaxes(title="time steps")
    fig.update_yaxes(title="log_div/dt")
    fig.update_layout(margin=dict(t=50))

    fig.update_layout(bargap=0.0)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01)
        )

    st.plotly_chart(fig)


def st_all_data_measures(data_dict: dict[str, np.ndarray], dt: float = 1.0, key: str | None = None
                         ) -> None:
    """Streamlit element for all data measures.

    Args:
        data_dict: Dictionary containing the time series.
        dt: The time step.
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    if st.checkbox("Consecutive extrema", key=f"{key}__st_all_data_measures__ce"):
        st.markdown("**Plot consecutive minima or maxima for individual dimensions:**")
        st_extrema_map(data_dict, key=f"{key}__st_all_data_measures")
    utils.st_line()
    if st.checkbox("Statistical measures", key=f"{key}__st_all_data_measures__sm"):
        st.markdown("**Plot the standard deviation, variance, mean or median of the time series:**")
        st_statistical_measures(data_dict, key=f"{key}__st_all_data_measures")
    utils.st_line()
    if st.checkbox("Histogram", key=f"{key}__st_all_data_measures__hist"):
        st.markdown("**Plot the value histogram for individual dimensions:**")
        st_histograms(data_dict, key=f"{key}__st_all_data_measures")
    utils.st_line()
    if st.checkbox("Power spectrum", key=f"{key}__st_all_data_measures__ps"):
        st.markdown("**Plot the mean or dimension resolved power spectrum:**")
        st_power_spectrum(data_dict, dt=dt, key=f"{key}__st_all_data_measures")
    utils.st_line()
    if st.checkbox("Lyapunov from data", key=f"{key}__st_all_data_measures__ledata"):

        st.markdown("**Plot the logarithmic trajectory divergence from data.**")
        with st.expander("More info ..."):
            st.write("The algorithm is based on the Rosenstein algorithm. Original Paper: Rosenstein et. al. (1992).")
            st.write("The sloap of the linear fit represents the largest Lyapunov exponent.")
        st_largest_lyapunov_from_data(data_dict, dt=dt, key=f"{key}__st_all_data_measures")


@st.experimental_memo
def get_error(y_pred_traj: np.ndarray,
              y_true_traj: np.ndarray, ) -> np.ndarray:
    """Get the error between y_pred_traj and y_true_traj.
    TODO: Not good that only one normalization is hardcoded.
    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.

    Returns:
        The error over time of shape (time_steps, ).
    """
    error_series = meas.error_over_time(y_pred_traj, y_true_traj,
                                        normalization="root_of_avg_of_spacedist_squared")
    return error_series


@st.experimental_memo
def get_valid_time_index(error_series: np.ndarray, error_threshold: float, ) -> int:
    """Get the valid time index from an error_series.

    Args:
        error_series: The error over time of shape (time_steps, ).
        error_threshold: The error threshold.

    Returns:
        The valid time index as an integer.
    """
    return meas.valid_time_index(error_series=error_series, error_threshold=error_threshold)


def st_show_error(y_pred_traj: np.ndarray,
                  y_true_traj: np.ndarray) -> None:
    """Streamlit element to show the error between a prediction and a true time series.

    TODO: root_of_avg_of_spacedist_squared is the only normalization available at the moment .

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.

    """
    error = get_error(y_pred_traj, y_true_traj)
    data_dict = {"Error": error}
    figs = plpl.multiple_1d_time_series(data_dict, y_label="Error")
    plpl.multiple_figs(figs)


def st_show_valid_times_vs_error_threshold(y_pred_traj: np.ndarray,
                                           y_true_traj: np.ndarray,
                                           dt: float,
                                           key: str | None = None) -> None:
    """Streamlit element to show a valid times vs error threshold plot.

    # TODO: Decide whether the session state stuff is too complicated.
    # TODO: Add valid time for 0.4 to streamlit session state.

    There is an option to load the last measured value of st_largest_lyapunov_exponent with
    streamlit session state variables.

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.
        dt: The time step.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    error_series = get_error(y_pred_traj, y_true_traj)
    error_thresh_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresh_steps = len(error_thresh_list)
    valid_times = np.zeros(thresh_steps)
    for i, thresh in enumerate(error_thresh_list):
        valid_time = get_valid_time_index(error_series, error_threshold=thresh)
        valid_times[i] = valid_time

    time_axis = st.selectbox("Time axis", ["steps", "real time", "lyapunov times"],
                             key=f"{key}__st_show_valid_times_vs_error_threshold__ta")

    if time_axis == "steps":
        y_label_add = "steps"
    elif time_axis == "real time":
        y_label_add = "real time"
        valid_times *= dt
    elif time_axis == "lyapunov times":

        latest_measured_lle = utils.st_get_session_state(prefix=STATE_PREFIX, name="LLE")
        if latest_measured_lle is None:
            disabled = True
        else:
            disabled = False
        if st.button("Get latest measured LLE", disabled=disabled):
            default_lle = latest_measured_lle
        else:
            default_lle = 0.5

        lle = st.number_input(f"Largest Lyapunov exponent", value=default_lle,
                              min_value=0.001,
                              format="%f",
                              key=f"{key}__st_show_valid_times_vs_error_threshold__lle")

        y_label_add = "lyapunov time"
        valid_times *= dt * lle
    else:
        raise ValueError(f"This time_axis option {time_axis} is not accounted for.")

    data_dict = {"Valid time vs. thresh": valid_times}
    figs = plpl.multiple_1d_time_series(data_dict, y_label=f"Valid times in {y_label_add}",
                                        x_label="error threshold", x_scale=1/10)
    plpl.multiple_figs(figs)


def st_all_difference_measures(y_pred_traj: np.ndarray,
                               y_true_traj: np.ndarray,
                               dt: float,
                               key: str | None = None
                               ) -> None:
    """Streamlit element for all difference based measures.

    - Plots the difference y_true - y_pred.
    - Plots the error(y_true, y_pred).
    - Plots the valid time vs. error threshold.

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.
        dt: The time step.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    if st.checkbox("Trajectory", key=f"{key}__st_all_difference_measures__traj"):
        difference_dict = {"Difference": y_true_traj - y_pred_traj}
        figs = plpl.multiple_1d_time_series(difference_dict,
                                            subplot_dimensions_bool=False,
                                            y_label="true - fit")
        plpl.multiple_figs(figs)
    utils.st_line()
    if st.checkbox("Error", key=f"{key}__st_all_difference_measures__error"):
        st_show_error(y_pred_traj, y_true_traj)
    utils.st_line()
    if st.checkbox("Valid time", key=f"{key}__st_all_difference_measures__vt"):
        st_show_valid_times_vs_error_threshold(y_pred_traj, y_true_traj, dt=dt)


if __name__ == "__main__":
    pass
