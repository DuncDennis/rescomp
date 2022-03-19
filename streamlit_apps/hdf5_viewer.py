import streamlit as st
import pathlib
import h5py
import os
import matplotlib.pyplot as plt
import rescomp.plotting as plot
import rescomp.statistical_tests as stattest


# @st.cache
def load_hdf5_file(file_path):
    f = h5py.File(file_path, 'r')
    return f


def check_parameter_sweep(f):
    pass


repo_path = pathlib.Path(__file__).parent.resolve().parents[0]
path = pathlib.Path.joinpath(repo_path, "results")
print(path)

hdf5_files = [x for x in os.listdir(path) if x.endswith(".hdf5")]

option = st.sidebar.selectbox(
    'Select experiment', hdf5_files)

if option is not None:
    f = load_hdf5_file(pathlib.Path.joinpath(path, option))
    # st.sidebar.write('You selected:', f"{f.keys()}")

    runs = list(f["runs"].keys())

    option2 = st.sidebar.selectbox(
        'runs', runs)

    for key, val in f["runs"][option2].attrs.items():
        st.sidebar.write(f"{key}: {val}")

    data = f["runs"][option2][:]
    st.write(data.shape)


    # fig, ax = plt.subplots()
    # ax.plot(data[0, 0, 0, :, 0], data[0, 0, 0, :, 2], linewidth=1, label="pred")
    # ax.plot(data[0, 0, 1, :, 0], data[0, 0, 1, :, 2], linewidth=1, label="true")
    # ax.set_xlim(-20, 20)
    # ax.set_ylim(5, 45)
    # plt.legend()
    # ax.plot(y_pred[:, 0], y_pred[:, 2], linewidth=1, label="pred")

    # fig = plot.plot_attractor(data)
    # st.pyplot(fig)

# st.button("test")
