import streamlit as st

import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.app_fragments.measures as measures
import streamlit_project.app_fragments.utils as utils
import streamlit_project.app_fragments.plotting as plot
import streamlit_project.app_fragments.esn as esn


if __name__ == '__main__':
    st.set_page_config("Basic ESN Viewer", page_icon="🥦")

    st.header("Basic ESN Viewer")

    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = syssim.st_select_system()

        t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred = \
            syssim.st_select_time_steps_split_up(default_t_train_disc=2500,
                                                 default_t_train_sync=200,
                                                 default_t_train=5000,
                                                 default_t_pred_disc=2500,
                                                 default_t_pred_sync=200,
                                                 default_t_pred=1000)
        section_steps = [t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred]
        section_names = ["train disc", "train sync", "train", "pred disc", "pred sync", "pred"]

        time_steps = sum(section_steps)

        if "dt" in system_parameters.keys():
            dt = system_parameters["dt"]
        else:
            dt = 1.0

        time_series = syssim.simulate_trajectory(system_name, system_parameters, time_steps)
        time_series = syssim.st_preprocess_simulation(time_series)
        x_dim = time_series.shape[1]

        utils.st_line()

    with st.sidebar:
        st.header("ESN: ")
        esn_type, basic_build_args = esn.st_basic_esn_build()
        st.subheader("System specific parameters: ")
        with st.expander("Network parameters: "):
            build_args = basic_build_args | esn.st_network_build_args()
        utils.st_line()

    with st.sidebar:
        st.header("Seed: ")
        utils.st_line()
        seed = 1  # TODO: add seed handling.

    with st.expander("Input data"):
        utils.st_line()
        if st.checkbox("Plot"):
            plot.st_default_simulation_plot(time_series)
        utils.st_line()
        if st.checkbox("Sections"):
            plot.st_one_dim_time_series_with_sections(time_series,
                                                      section_steps=section_steps,
                                                      section_names=section_names)
    with st.expander("Build and train"):
        if st.checkbox("Build and train"):
            esn_obj = esn.build(esn_type, seed=seed, x_dim=x_dim, **build_args)

    with st.expander("Predict"):
        pass
