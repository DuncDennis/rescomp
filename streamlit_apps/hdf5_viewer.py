import streamlit as st
import pathlib
import h5py
import os
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np
import pandas as df
import rescomp.plotting as plot
import rescomp.statistical_tests as stattest


# @st.cache
def load_hdf5_file(file_path):
    f = h5py.File(file_path, 'r')
    return f


def check_parameter_sweep(f, stripped=True):
    params_dict = {}
    for trajectory in list(f["runs"].keys()):
        for key, val in f["runs"][trajectory].attrs.items():
            if key in params_dict.keys():
                if stripped:
                    if val not in params_dict[key]:
                        params_dict[key].append(val)
                else:
                    params_dict[key].append(val)
            else:
                params_dict[key] = [val, ]
    return params_dict


def get_trajectories(multiselect_dict, f):
    for key, val in multiselect_dict.items():
        if len(val) == 0:
            st.warning(f"Choose at least one option for {key}")
            raise Exception(f"Choose at least one option for {key}")

    trajs = []
    params_to_show = []
    for trajectory in list(f["runs"].keys()):
        select_traj = True
        for key, val in f["runs"][trajectory].attrs.items():
            if val not in multiselect_dict[key]:
                select_traj = False
                break
        if select_traj:
            params_this_traj = {}
            trajs.append(trajectory)
            for key, val in multiselect_dict.items():
                if len(val)>1:
                    params_this_traj[key] = f["runs"][trajectory].attrs[key]
            params_to_show.append(params_this_traj)
    return trajs, params_to_show


repo_path = pathlib.Path(__file__).parent.resolve().parents[0]
path = pathlib.Path.joinpath(repo_path, "results")
print(path)

hdf5_files = [x for x in os.listdir(path) if x.endswith(".hdf5")]

option = st.sidebar.selectbox(
    'Select experiment', hdf5_files)

if option is not None:
    f = load_hdf5_file(pathlib.Path.joinpath(path, option))

    params_dict = check_parameter_sweep(f)

    multiselect_dict = {}
    for key, val in params_dict.items():
        multiselect_dict[key] = st.sidebar.multiselect(key, val)

    # st.write(multiselect_dict)
    trajs, params_to_show = get_trajectories(multiselect_dict, f)

    with st.container():
        plot_vt = st.checkbox("plot valid times heatmap")
        if plot_vt:
            error_threshhold = st.slider("error threshhold", min_value=0.00001, max_value=3., step=0.001, value=0.5)
            fig = plot.plot_valid_times_heatmap(trajs, params_to_show, f, error_threshhold=error_threshhold)
            st.pyplot(fig)

    with st.container():
        plot_error = st.checkbox("plot error")
        if plot_error:
            max_x = st.slider("max_x", min_value=5, max_value=1000, step=1, value=500)
            fig = plot.plot_error(trajs, params_to_show, f, max_x)
            st.pyplot(fig)

    with st.container():
        plot_difference = st.checkbox("plot difference")
        if plot_difference:
            fig = plot.plot_difference(trajs, params_to_show, f)
            st.pyplot(fig)

    plot_attractor = st.checkbox("plot attractor")
    plot_trajectory = st.checkbox("plot trajectory")
    if plot_attractor or plot_trajectory:
        N_ens = f["runs"][trajs[0]][:].shape[0]
        i_ens = st.number_input("i_ens", min_value=0, max_value=N_ens-1)
        N_time_periods = f["runs"][trajs[0]][:].shape[1]
        i_time_period = st.number_input("i_time_period", min_value=0, max_value=N_time_periods-1)

        if plot_attractor:
            fig = plot.plot_attractor_2(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(15, 4))
            st.pyplot(fig)

        if plot_trajectory:
            fig = plot.plot_trajectories(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(15, 4))
            st.pyplot(fig)

