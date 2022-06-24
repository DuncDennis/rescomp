import streamlit as st
import pathlib
import h5py
import os

import rescomp.plotting as plot

import matplotlib.pyplot as plt
plt.style.use('dark_background')


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


repo_path = pathlib.Path(__file__).parent.resolve().parents[1]
path = pathlib.Path.joinpath(repo_path, "results/w_out_pca_tests")
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
    with st.sidebar:
        st.write(f"N_ens: {N_ens}")

    params_dict = check_parameter_sweep(f)

    multiselect_dict = {}
    with st.sidebar:
        for key, val in params_dict.items():
            disabled = True if len(val) == 1 else False
            multiselect_dict[key] = st.multiselect(key, val, default=val[0], disabled=disabled)

    trajs, params_to_show = get_trajectories(multiselect_dict, f)

    with st.expander("W_out_mean_index vs. Parameter"):
        w_out_mean_index_vs_parameter = st.checkbox("W_out_mean_index vs. Params")
        if w_out_mean_index_vs_parameter:
            st.header("W_out_mean_index vs. Parameter:")

            multiselect_dict_only_multiples = {key: val for key, val in multiselect_dict.items() if len(val)>1}

            sweep_variable = st.selectbox("sweep variable", multiselect_dict_only_multiples.keys())

            log_x = st.checkbox("log_x")
            average_type = st.selectbox("average_type", ["mean", "median"])

            fig = plot.plot_w_out_index_sweep(trajs, params_to_show, f, sweep_variable, figsize=(800, 500), log_x=log_x,
                                              average_type=average_type)
            st.plotly_chart(fig)

    with st.expander("W_out_mean_distribution vs. Parameter"):
        w_out_mean_distribution = st.checkbox("W_out_mean_distribution vs. Params")
        if w_out_mean_distribution:
            st.header("W_out_mean_distribution vs. Parameter:")
            print(trajs)
            for i_traj, traj in enumerate(trajs):
                data = f["runs"][traj][:]
                figs = plot.plot_w_out_mean_distribution(data)
                for fig in figs:
                    st.plotly_chart(fig)
                print(data.shape)
            # multiselect_dict_only_multiples = {key: val for key, val in multiselect_dict.items() if len(val)>1}

            # sweep_variable = st.selectbox("sweep variable", multiselect_dict_only_multiples.keys())
            #
            # log_x = st.checkbox("log_x")
            # average_type = st.selectbox("average_type", ["mean", "median"])
            #
            # fig = plot.plot_w_out_index_sweep(trajs, params_to_show, f, sweep_variable, figsize=(800, 500), log_x=log_x,
            #                                   average_type=average_type)
            # st.plotly_chart(fig)
