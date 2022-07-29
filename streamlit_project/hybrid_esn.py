import numpy as np
import streamlit as st

import rescomp
import rescomp.plotting as plot
import rescomp.statistical_tests as stat_test
import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.generalized_plotting.plotly_plots as plpl
import streamlit_project.generalized_plotting.temp_plotly_plots as tplpl
import streamlit_project.app_fragments.esn_build_temp as esn_app
import streamlit_project.app_fragments.utilities_temp as util_app


if __name__ == '__main__':

    st.set_page_config("Hybrid-ESN App", page_icon=":shark")

    with st.sidebar:
        st.header("Real system: ")

        system_name, system_parameters = syssim.st_select_system()

        if "dt" in system_parameters.keys():
            dt_bool = True
        else:
            dt_bool = False

        t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred = \
            syssim.st_select_time_steps_split_up(default_t_train_disc=2500,
                                                 default_t_train_sync=200,
                                                 default_t_train=5000,
                                                 default_t_pred_disc=2500,
                                                 default_t_pred_sync=200,
                                                 default_t_pred=1000)
        time_steps = (t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred)

    with st.sidebar:
        st.header("Architecture: ")
        esn_type, build_args = esn_app.st_basic_esn_build(("ESN_normal",
                                                           "ESN_output_hybrid",
                                                           "ESN_input_hybrid",
                                                           "ESN_full_hybrid"))

        st.subheader("System specific parameters: ")
        with st.expander("Network parameters: "):
            build_args = build_args | esn_app.st_network_build_args()

        if "hybrid" in esn_type:
            with st.expander("Hybrid parameters: "):
                build_args = build_args | esn_app.st_hybrid_build_args(esn_type, system_name,
                                                                       system_parameters)
        print(build_args)

    with st.sidebar:
        st.header("Seed: ")
        seed = util_app.seed_interface()

    # INFO:
    st.header("Info: ")
    ct = st.container()
    util_app.line()

    with st.expander("Simulate Trajectory: "):
        simulated_bool = st.checkbox("Simulate Trajectory: ")
        if simulated_bool:
            time_series = syssim.simulate_trajectory(system_name, system_parameters, time_steps)
            x_dim = time_series.shape[1]

            if st.checkbox("Plot: "):
                syssim.st_default_simulation_plot(time_series)

            time_boundaries = (0, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred)
            left, right = st.columns(2)
            with left:
                plot_trajectory_time_series = st.checkbox("Plot one dimension: ")
            with right:
                i_dim = st.number_input("i_dim", min_value=0, max_value=x_dim - 1)
            if plot_trajectory_time_series:
                fig = tplpl.plot_1d_time_series(time_series, i_dim, boundaries=time_boundaries)
                st.plotly_chart(fig, use_container_width=True)

            ct.metric("r_dim to x_dim factor: ", value=np.round(build_args["r_dim"]/x_dim, 1))

    disabled = False if simulated_bool else True
    with st.expander("RC architecture"):
        build_bool = st.checkbox("Build RC architecture", disabled=disabled)
        if build_bool:
            esn = esn_app.build(esn_type, seed, x_dim=x_dim, **build_args)
            print(esn)

    disabled = False if build_bool else True
    with st.expander("Train RC"):
        train_bool = st.checkbox("Train RC", disabled=disabled)
        if train_bool:
            x_train, x_pred_list = stat_test.data_simulation(time_series, t_train_disc,
                                                             t_train_sync, t_train,
                                                             t_pred_disc, t_pred_sync, t_pred,
                                                             nr_of_time_intervals=1,
                                                             sim_data_return=False, v=0)
            esn, x_train_true, x_train_pred = esn_app.train(
                esn,
                x_train,
                t_train_sync)

            if st.checkbox("Show training: "):
                data = {"train true": x_train_true,
                        "train predict": x_train_pred}
                data_diff = {"diff": x_train_true - x_train_pred}
                if x_dim > 3:
                    figs = plpl.multiple_time_series_image(data | data_diff)
                    plpl.multiple_figs(figs)
                elif x_dim == 3:
                    fig = plpl.multiple_3d_time_series(data)
                    st.plotly_chart(fig)

                    figs = plpl.multiple_1d_time_series(data_diff, subplot_dimensions_bool=False)
                    plpl.multiple_figs(figs)

                elif x_dim == 2:
                    mode = "line" if dt_bool else "scatter"
                    fig = plpl.multiple_2d_time_series(data, mode=mode)
                    st.plotly_chart(fig)

                    figs = plpl.multiple_1d_time_series(data_diff, subplot_dimensions_bool=False)
                    plpl.multiple_figs(figs)

                elif x_dim == 1:
                    figs = plpl.multiple_1d_time_series(data)
                    plpl.multiple_figs(figs)

                    figs = plpl.multiple_1d_time_series(data_diff, subplot_dimensions_bool=False)
                    plpl.multiple_figs(figs)

    disabled = False if train_bool else True
    with st.expander("Prediction"):
        predict_bool = st.checkbox("Predict with RC", disabled=disabled)
        if predict_bool:
            x_pred = x_pred_list[0]
            y_pred, y_true = esn_app.predict(esn, x_pred, t_pred_sync)

            if st.checkbox("Show prediction: "):

                data = {"true": y_true,
                        "pred": y_pred,
                        }
                data_diff = {"diff": y_true - y_pred}
                if x_dim > 3:
                    figs = plpl.multiple_time_series_image(data | data_diff)
                    plpl.multiple_figs(figs)
                elif x_dim == 3:
                    fig = plpl.multiple_3d_time_series(data)
                    st.plotly_chart(fig)

                    figs = plpl.multiple_1d_time_series(data_diff, subplot_dimensions_bool=False)
                    plpl.multiple_figs(figs)

                elif x_dim == 2:
                    mode = "line" if dt_bool else "scatter"
                    fig = plpl.multiple_2d_time_series(data, mode=mode)
                    st.plotly_chart(fig)

                    figs = plpl.multiple_1d_time_series(data_diff, subplot_dimensions_bool=False)
                    plpl.multiple_figs(figs)

                elif x_dim == 1:
                    figs = plpl.multiple_1d_time_series(data)
                    plpl.multiple_figs(figs)

                    figs = plpl.multiple_1d_time_series(data_diff, subplot_dimensions_bool=False)
                    plpl.multiple_figs(figs)

            left, right = st.columns(2)
            with left:
                valid_times_show = st.checkbox("Show valid_time vs error_threshhold: ")

                if valid_times_show:
                    disabled = False
                else:
                    disabled = True
            with right:
                if st.checkbox("In lyapunov times", disabled=disabled):
                    lle = st.number_input("LLE", value=0.9, format="%f", min_value=0.0)

                    if dt_bool:
                        dt = system_parameters["dt"]
                    else:
                        dt = 1.0
                    in_lyapunov_times = {"dt": dt, "LE": lle}

                else:
                    in_lyapunov_times = None

            if valid_times_show:
                fig = tplpl.plot_valid_times_vs_pred_error(y_pred, y_true,
                                                           in_lyapunov_times=in_lyapunov_times)
                st.plotly_chart(fig)
