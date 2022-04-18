import streamlit as st
import pathlib
import h5py
import os

import rescomp.plotting as plot

import matplotlib.pyplot as plt
plt.style.use('dark_background')
# import rescomp.statistical_tests as stattest


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

# hdf5_files = [x for x in os.listdir(path) if x.endswith(".hdf5")]
hdf5_files = sorted(pathlib.Path(path).iterdir(), key=os.path.getmtime)[::-1]
hdf5_files = [x.name.split(".")[0] for x in hdf5_files if x.name.endswith(".hdf5")]

with st.sidebar:
    option = st.sidebar.selectbox(
        'Select experiment', hdf5_files)
    st.markdown("""---""")

# option_best = st.sidebar.button("Get best Parameters")
# TODO: Add this functionality

if option is not None:
    f = load_hdf5_file(pathlib.Path.joinpath(path, option+".hdf5"))
    data_shape = f["runs"]["trajectory_1"][:].shape
    N_ens = data_shape[0]
    N_time_periods = data_shape[1]
    with st.sidebar:

        st.write(f"N_ens: {N_ens}")
        st.sidebar.write(f"N_time_periods: {N_time_periods}")

    params_dict = check_parameter_sweep(f)

    multiselect_dict = {}
    with st.sidebar:
        for key, val in params_dict.items():
            disabled = True if len(val) == 1 else False
            multiselect_dict[key] = st.multiselect(key, val, default=val[0], disabled=disabled)

    trajs, params_to_show = get_trajectories(multiselect_dict, f)

    data_shape = f["runs"][trajs[0]][:].shape

    with st.container():
        plot_vt = st.checkbox("plot valid times heatmap")
        if plot_vt:
            error_threshhold = st.slider("error threshhold", min_value=0.00001, max_value=3., step=0.001, value=0.5)
            fig = plot.plot_valid_times_heatmap(trajs, params_to_show, f, error_threshhold=error_threshhold)
            st.pyplot(fig)
        st.markdown("""---""")

    with st.container():
        plot_error = st.checkbox("plot error")

        if plot_error:
            st.header("Plot Error over Time:")

            left, right = st.columns(2)
            with left:
                error_bar = st.checkbox("show error bars")
            with right:
                ylog = st.checkbox('y log plot')

            max_value = f["runs"][trajs[0]][:].shape[-2]
            max_x = st.slider("max_x", min_value=5, max_value=max_value, step=1, value=500)
            fig = plot.plot_error(trajs, params_to_show, f, max_x, error_bar=error_bar, ylog=ylog)
            st.pyplot(fig)
        st.markdown("""---""")

    left, mid, right = st.columns(3)
    with left:
        plot_attractor = st.checkbox("plot attractor")
    with mid:
        plot_trajectory = st.checkbox("plot trajectory")
    with right:
        plot_corrdim = st.checkbox("plot correlation dimension")
    if plot_attractor or plot_trajectory or plot_corrdim:


        left, right = st.columns(2)
        with left:
            i_ens = st.number_input("i_ens", min_value=0, max_value=N_ens-1)
        with right:
            i_time_period = st.number_input("i_time_period", min_value=0, max_value=N_time_periods-1)

        if plot_attractor:
            st.header("Plot Attractor: ")
            fig = plot.plot_attractor_2(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(15, 4))
            st.pyplot(fig)

        if plot_trajectory:
            N_dims = f["runs"][trajs[0]][:].shape[-1]
            st.header("Plot Trajectory: ")
            i_dim = st.number_input("dimension", min_value=0, max_value=N_dims-1)
            fig = plot.plot_trajectories(trajs, params_to_show, f, i_ens, i_time_period, i_dim, base_fig_size=(15, 4))
            st.pyplot(fig)

        if plot_corrdim:
            st.header("Plot Correlation Dimension")
            nr_steps = st.slider("Nr steps for correlation dim", min_value=1, max_value=600, step=1, value=10)
            fig = plot.plot_correlation_dimension(trajs, params_to_show, f, i_ens, i_time_period, figsize=(8, 4),
                                                  nr_steps=nr_steps)
            st.pyplot(fig)
    st.markdown("""---""")

    with st.container():
        plot_corrdim_hist = st.checkbox("plot correlation dimension HISTOGRAM")
        if plot_corrdim_hist:
            st.header("Plot Correlation Dimension Histogram:")
            nr_steps = st.slider("Nr steps for correlation dim (hist)", min_value=1, max_value=600, step=1, value=10)
            bins = st.slider("histogram bins", min_value=1, max_value=100, step=1, value=10)
            fig = plot.plot_correlation_dimension_hist(trajs, params_to_show, f, nr_steps=nr_steps,
                                                       base_figsize=(15, 4), bins=bins)
            st.pyplot(fig)

