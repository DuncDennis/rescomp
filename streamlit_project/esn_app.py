import streamlit as st

import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.app_fragments.measures as measures
import streamlit_project.app_fragments.utils as utils
import streamlit_project.app_fragments.plotting as plot
import streamlit_project.app_fragments.esn as esn


if __name__ == '__main__':
    st.set_page_config("Basic ESN Viewer", page_icon="⚡")

    with st.sidebar:
        st.header("ESN Viewer")
        simulate_bool, build_bool, train_bool, predict_bool = utils.st_main_checkboxes()

        utils.st_line()
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

        scale, shift, noise_scale = syssim.st_preprocess_simulation()

        if simulate_bool:
            time_series = syssim.simulate_trajectory(system_name, system_parameters, time_steps)
            time_series = syssim.preprocess_simulation(time_series,
                                                       scale=scale,
                                                       shift=shift,
                                                       noise_scale=noise_scale)
            time_series_dict = {"time series": time_series}

            x_train, x_pred = syssim.split_time_series_for_train_pred(time_series,
                                                                      t_train_disc=t_train_disc,
                                                                      t_train_sync=t_train_sync,
                                                                      t_train=t_train,
                                                                      t_pred_disc=t_pred_disc,
                                                                      t_pred_sync=t_pred_sync,
                                                                      t_pred=t_pred,
                                                                      )
            x_dim = time_series.shape[1]
        utils.st_line()

    with st.sidebar:
        st.header("ESN: ")
        esn_type = esn.st_select_esn_type()
        with st.expander("Basic parameters: "):
            basic_build_args = esn.st_basic_esn_build()
        with st.expander("Network parameters: "):
            build_args = basic_build_args | esn.st_network_build_args()
        utils.st_line()

    with st.sidebar:
        st.header("Seed: ")
        seed = utils.st_seed()
        utils.st_line()

    sim_data_tab, tab2, tab3, tab4, tab5 = st.tabs(
        ["🌀 Simulated data", "🛠️ Architecture", "🦾 Training",
         "🔮 Prediction", "🔬 Other visualizations"])

    with sim_data_tab:
        if simulate_bool:
            # TODO: add it all to a streamlit element function?
            st.info("Plot the simulated data and measure some quantites.")
            plot_tab, measure_tab, train_pred_tab, lyapunov_tab = st.tabs(["Plot", "Measures",
                                                                           "Train-predict-split",
                                                                           "Lyapunov Exponent"])
            with plot_tab:
                plot.st_all_plots(time_series_dict, key="simulation")

            with measure_tab:
                measures.st_all_data_measures(time_series_dict, dt=dt, key="simulation")

            with train_pred_tab:
                if st.checkbox("Train / predict split"):
                    plot.st_one_dim_time_series_with_sections(time_series,
                                                              section_steps=section_steps,
                                                              section_names=section_names)

            with lyapunov_tab:
                if st.checkbox("Calculate Lyapunov exponent of system"):
                    measures.st_largest_lyapunov_exponent(system_name, system_parameters)

        else:
            st.info('Activate [🌀 Simulate data] checkbox to see something.')

    with tab2:
        if build_bool:
            st.info("TBD: Some architecture plots.")
            esn_obj = esn.build(esn_type, seed=seed, x_dim=x_dim, **build_args)
            if st.checkbox("W_in matrix"):
                st.write(esn_obj._w_in)
        else:
            st.info('Activate [🛠️ Build] checkbox to see something.')

    with tab3:
        if train_bool:
            y_train_fit, y_train_true = esn.train(esn_obj, x_train, t_train_sync)
            train_data_dict = {"train true": y_train_true,
                               "train fitted": y_train_fit}
            plot_tab, measure_tab = st.tabs(["Plot", "Measures"])

            with plot_tab:
                plot.st_all_plots(train_data_dict, key="train")
            with measure_tab:
                measures.st_all_data_measures(train_data_dict, dt=dt, key="train")

        else:
            st.info('Activate [🦾 Train] checkbox to see something.')

    with tab4:
        if predict_bool:
            y_pred, y_pred_true = esn.predict(esn_obj, x_pred, t_pred_sync)
            if st.checkbox("Show prediction"):
                pred_data = {"true": y_pred_true,
                             "pred": y_pred}
                # train_data_diff = {"Difference": y_pred_true - y_pred}

                plot.st_plot_dim_selection(pred_data, key="pred")
        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')

    with tab5:
        st.write("TBD")

    # with st.expander("🛠️ Build"):
    #     utils.st_line()
    #     if build_bool:
    #         utils.st_line()
    #         esn_obj = esn.build(esn_type, seed=seed, x_dim=x_dim, **build_args)
    #         # st.write(esn_obj._w_out)
    #
    #         y_train_fit, y_train_true = esn.train(esn_obj, x_train, t_train_sync)
    #         # st.write(esn_obj._w_out)
    # with st.expander("🦾 Train"):
    #     if st.checkbox("Show training"):
    #         train_data = {"train true": y_train_true,
    #                       "train fitted": y_train_fit}
    #         # train_data_diff = {"Difference": y_train_true - y_train_fit}
    #
    #         plot.st_plot_dim_selection(train_data, key="train")
    #
    # with st.expander("🔮 Predict"):
    #     utils.st_line()
    #     disabled = False if train_bool else True
    #     predict_bool = st.checkbox("Predict", disabled=disabled)
    #     if predict_bool:
    #         utils.st_line()
    #         y_pred, y_pred_true = esn.predict(esn_obj, x_pred, t_pred_sync)
    #
    #         if st.checkbox("Show prediction"):
    #             pred_data = {"true": y_pred_true,
    #                          "pred": y_pred}
    #             # train_data_diff = {"Difference": y_pred_true - y_pred}
    #
    #             plot.st_plot_dim_selection(pred_data, key="pred")
