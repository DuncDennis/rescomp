"""A streamlit app to demonstrate PCA is conjunction with Echo State Networks - ESN and PCA."""
import copy
import streamlit as st

import streamlit_project.app_fragments.esn_app_utilities as esnutils
import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.app_fragments.timeseries_measures as measures
import streamlit_project.app_fragments.pred_vs_true_plotting as pred_vs_true
import streamlit_project.app_fragments.system_measures as sysmeas
import streamlit_project.app_fragments.streamlit_utilities as utils
import streamlit_project.app_fragments.timeseries_plotting as plot
import streamlit_project.app_fragments.esn_build_train_predict as esn
import streamlit_project.app_fragments.esn_plotting as esnplot

import numpy as np
import plotly.express as px
import streamlit_project.generalized_plotting.plotly_plots as plpl
import rescomp.measures_new as rescompmeasures
if __name__ == '__main__':

    st.set_page_config("ESN and PCA", page_icon="💫")

    with st.sidebar:
        st.header("ESN Viewer")
        utils.st_reset_all_check_boxes()

        simulate_bool, build_bool, train_bool, predict_bool = esnutils.st_main_checkboxes()

        utils.st_line()
        st.header("System: ")
        system_name, system_parameters = syssim.st_select_system()

        t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred = \
            syssim.st_select_time_steps_split_up(default_t_train_disc=2500,
                                                 default_t_train_sync=200,
                                                 default_t_train=5000,
                                                 default_t_pred_disc=2500,
                                                 default_t_pred_sync=200,
                                                 default_t_pred=3000)
        section_steps = [t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred]
        section_names = ["train disc", "train sync", "train", "pred disc", "pred sync", "pred"]

        time_steps = sum(section_steps)

        if "dt" in system_parameters.keys():
            dt = system_parameters["dt"]
        else:
            dt = 1.0

        shift_scale_params, noise_scale = syssim.st_preprocess_simulation()
        utils.st_line()

    with st.sidebar:
        st.header("ESN: ")
        esn_type = esn.st_select_esn_type(("ESN_strong", ))
        with st.expander("Basic parameters: "):
            basic_build_args = esn.st_basic_esn_build()
        with st.expander("Network parameters: "):
            build_args = basic_build_args | esn.st_network_build_args()
        with st.expander("Strong ESN parameters: "):
            build_args = build_args | esn.st_esn_strong_args()
        utils.st_line()

    with st.sidebar:
        st.header("Seed: ")
        seed = utils.st_seed()
        utils.st_line()

    sim_data_tab, build_tab, train_tab, predict_tab, other_vis_tab, esn_pca_tab = st.tabs(
        ["🌀 Simulated data",
         "🛠️ Architecture",
         "🦾 Training",
         "🔮 Prediction",
         "🔬 Look-under-hood",
         "💫 ESN PCA SECTION"])

    with sim_data_tab:
        if simulate_bool:

            time_series = syssim.simulate_trajectory(system_name, system_parameters,
                                                     time_steps)
            time_series, scale_shift_vector = syssim.preprocess_simulation(time_series,
                                                                           seed,
                                                                           scale_shift_params=shift_scale_params,
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

            st.markdown(
                "Plot and measure the **simulated data**, see which intervals are used for "
                "**training and prediction** and determine the **Lyapunov exponent** of the "
                "system. ")
            with st.expander("Show system equation: "):
                st.markdown(f"**{system_name}**")
                syssim.st_show_latex_formula(system_name)

            plot_tab, measure_tab, train_pred_tab, lyapunov_tab = st.tabs(["Plot", "Measures",
                                                                           "Train-predict-split",
                                                                           "Lyapunov Exponent"])
            with plot_tab:
                plot.st_all_timeseries_plots(time_series_dict, key="simulation")

            with measure_tab:
                measures.st_all_data_measures(time_series_dict, dt=dt, key="simulation")

            with train_pred_tab:
                if st.checkbox("Train / predict split"):
                    plot.st_one_dim_time_series_with_sections(time_series,
                                                              section_steps=section_steps,
                                                              section_names=section_names)

            with lyapunov_tab:
                if st.checkbox("Calculate Lyapunov exponent of system"):
                    sysmeas.st_largest_lyapunov_exponent(system_name, system_parameters)

        else:
            st.info('Activate [🌀 Simulate data] checkbox to see something.')

    with build_tab:
        if build_bool:
            esn_obj = esn.build_with_seed(esn_type, seed=seed, x_dim=x_dim, **build_args)
            esn_obj = copy.deepcopy(esn_obj)  # needed for the streamlit caching to work correctly.
            st.markdown("Explore the Echo State Network architecture.")
            tabs = st.tabs(["Dimensions", "Input matrix", "Network"])
            with tabs[0]:
                st.markdown("**Layer dimensions:**")
                architecture_container = st.container()
            with tabs[1]:
                w_in = esn_obj._w_in
                if st.checkbox("Input matrix as heatmap", key=f"build_tab__input_heatmap"):
                    esnplot.st_input_matrix_as_heatmap(w_in)
            with tabs[2]:
                network = esn_obj.return_network()
                esnplot.st_all_network_architecture_plots(network)

        else:
            st.info('Activate [🛠️ Build] checkbox to see something.')

    with train_tab:
        if train_bool:

            y_train_fit, y_train_true, res_train_dict, esn_obj = esn.train_return_res(esn_obj,
                                                                                      x_train,
                                                                                      t_train_sync,
                                                                                      )
            esn_obj = copy.deepcopy(esn_obj)  # needed for the streamlit caching to work correctly.
            train_data_dict = {"train true": y_train_true,
                               "train fitted": y_train_fit}
            st.markdown(
                "Compare the **training data** with the **fitted data** produced during training.")

            with st.expander("More info ..."):
                st.write(
                    "During training, the true training data and the fitted data should be very "
                    "similar. Otherwise the Echo State Network prediction is very likely to fail.")

            plot_tab, measure_tab, difference_tab = st.tabs(["Plot", "Measures", "Difference"])

            with plot_tab:
                plot.st_all_timeseries_plots(train_data_dict, key="train")
            with measure_tab:
                measures.st_all_data_measures(train_data_dict, dt=dt, key="train")
            with difference_tab:
                pred_vs_true.st_all_difference_measures(y_pred_traj=y_train_fit,
                                                        y_true_traj=y_train_true,
                                                        dt=dt,
                                                        train_or_pred="train",
                                                        key="train")
        else:
            st.info('Activate [🦾 Train] checkbox to see something.')

    with predict_tab:
        if predict_bool:

            y_pred, y_pred_true, res_pred_dict, esn_obj = esn.predict_return_res(esn_obj,
                                                                                 x_pred,
                                                                                 t_pred_sync)

            if scale_shift_vector is not None:
                if st.checkbox("Inverse scale and shift prediction: ", key="inv_scale_shift_pred"):
                    y_pred = syssim.inverse_transform_shift_scale(y_pred,
                                                                  scale_shift_params=shift_scale_params)
                    y_pred_true = syssim.inverse_transform_shift_scale(y_pred_true,
                                                                       scale_shift_params=shift_scale_params)

            esn_obj = copy.deepcopy(esn_obj)  # needed for the streamlit caching to work correctly.
            pred_data_dict = {"true": y_pred_true,
                              "pred": y_pred}
            st.markdown("Compare the Echo State Network **prediction** with the **true data**.")
            plot_tab, measure_tab, difference_tab = st.tabs(["Plot", "Measures", "Difference"])
            with plot_tab:
                plot.st_all_timeseries_plots(pred_data_dict, key="predict")
            with measure_tab:
                measures.st_all_data_measures(pred_data_dict, dt=dt, key="predict")
            with difference_tab:
                pred_vs_true.st_all_difference_measures(y_pred_traj=y_pred,
                                                        y_true_traj=y_pred_true,
                                                        dt=dt,
                                                        train_or_pred="predict",
                                                        key="predict")
        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')

    with other_vis_tab:
        if predict_bool:
            st.markdown("Explore internal quantities of the Echo State Network. ")

            res_states_tab, w_out_r_gen_tab, res_time_tab, res_dyn = st.tabs(
                ["Internal reservoir states", "W_out and R_gen",
                 "Reservoir time series", "Pure reservoir dynamics"])

            res_train_dict_no_rgen = {k: v for k, v in res_train_dict.items() if k != "r_gen"}
            res_pred_dict_no_rgen = {k: v for k, v in res_pred_dict.items() if k != "r_gen"}

            with res_states_tab:
                esnplot.st_reservoir_state_formula()

                if st.checkbox("Node value histograms"):
                    act_fct = esn_obj.get_act_fct()
                    esnplot.st_reservoir_states_histogram(res_train_dict_no_rgen,
                                                          res_pred_dict_no_rgen,
                                                          act_fct)
                utils.st_line()
                if st.checkbox("Node value time series", key=f"res_train_dict_no_rgen__checkbox"):
                    esnplot.st_reservoir_node_value_timeseries(res_train_dict_no_rgen,
                                                               res_pred_dict_no_rgen, )

                utils.st_line()
                if st.checkbox("Scatter matrix plot of reservoir states",
                               key="scatter_matrix_plot__checkbox"):
                    esnplot.st_scatter_matrix_plot(res_train_dict, res_pred_dict,
                                                   key="scatter_matrix_plot")

            with w_out_r_gen_tab:
                w_out = esn_obj.get_w_out()
                if st.checkbox("Output coupling", key="output_coupling_cb"):
                    st.markdown("Sum the absolute value of the W_out matrix over all generalized "
                                "reservoir indices, to see which output dimension has the "
                                "strongest coupling to the reservoir states.")
                    esnplot.st_plot_output_w_out_strength(w_out)
                utils.st_line()
                if st.checkbox("W_out matrix as barchart", key="w_out_as_bar"):
                    st.markdown(
                        "Show the w_out matrix as a stacked barchart, where the x axis is the "
                        "index of the generalized reservoir dimension.")
                    esnplot.st_plot_w_out_as_barchart(w_out)
                utils.st_line()
                if st.checkbox("R_gen std", key="r_gen_std"):
                    st.markdown(
                        "Show the standard deviation of the generalized reservoir state (r_gen) "
                        "during training and prediction.")
                    esnplot.st_r_gen_std_barplot(r_gen_train=res_train_dict["r_gen"],
                                                 r_gen_pred=res_pred_dict["r_gen"])
                utils.st_line()
                if st.checkbox("R_gen std times w_out", key="r_gen_std_wout"):
                    st.markdown(
                        "Show the standard deviation of the generalized reservoir state (r_gen) "
                        "times w_out during training and prediction.")
                    esnplot.st_r_gen_std_times_w_out_barplot(r_gen_train=res_train_dict["r_gen"],
                                                             r_gen_pred=res_pred_dict["r_gen"],
                                                             w_out=w_out)

            with res_time_tab:
                if st.checkbox("Reservoir states", key="r_states_3d"):
                    time_series_dict = {"r_train": res_train_dict["r"],
                                        "r_pred": res_pred_dict["r"]}
                    plot.st_timeseries_as_three_dim_plot(time_series_dict, key="r")
                utils.st_line()
                if st.checkbox("Generalized reservoir states", key="r_gen_states_3d"):
                    time_series_dict = {"r_gen_train": res_train_dict["r_gen"],
                                        "r_gen_pred": res_pred_dict["r_gen"]}
                    plot.st_timeseries_as_three_dim_plot(time_series_dict, key="r_gen")

            with res_dyn:
                if st.checkbox("Largest lyapunov exponent of reservoir", key="lle_res"):
                    st.markdown(
                        "Calculate the largest lyapunov exponent from the trained reservoir "
                        "update equation, looping the output back into the reservoir.")
                    # TODO: Say that the last training reservoir state is used.
                    # TODO: Add Latex formula for reservoir update equation.
                    res_update_func = esn_obj.get_res_iterator_func()
                    res_starting_point = res_train_dict["r"][-1, :]
                    sysmeas.st_largest_lyapunov_exponent_custom(res_update_func,
                                                                res_starting_point,
                                                                dt=dt,
                                                                using_str="the reservoir update "
                                                                          "equation")

        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')

    with esn_pca_tab:
        if predict_bool:

            st.markdown(
                r"""
                Select your preferred *Strong ESN parameters* on the right. 
                The following flow chart visualizes the settings: 
                """
            )
            with st.expander("Show flowchart: "):
                from PIL import Image
                image = Image.open('other_src/ESN_strong.jpg')
                st.image(image, caption='ESN strong')

            st.markdown(
                r"""
                Basically we can: 
                - Add noise to r_train and input_train
                - Perform PCA and use pca as regressors. 
                - Perform PCA to get a regularized w_out for the normal regressors. 
                """
            )

            states_to_save = esn_obj.states_to_save

            if st.checkbox("moment matrix"):
                r_gen_states = states_to_save["r_gen_states"]
                moment_matrix = r_gen_states.T @ r_gen_states
                with st.expander("Moment Matrix: "):
                    st.write(moment_matrix)
                rank = np.linalg.matrix_rank(moment_matrix)

                cond = np.linalg.cond(moment_matrix)
                cols = st.columns(2)
                with cols[0]:
                    st.write("Condition: ")
                    st.write(cond)
                with cols[1]:
                    st.write("Rank: ")
                    st.write(rank)

            if st.checkbox("Explained variance: "):
                if build_args["perform_pca_bool"]:
                    explained_variance = esn_obj._pca.explained_variance_
                    fig = px.line(y=explained_variance)
                    fig.add_vline(x=rank)
                    fig.update_yaxes(type="log", exponentformat="E")

                    st.plotly_chart(fig)

            if st.checkbox("Train on subset"):
                w_out_subsets = esn_obj.train_on_subset(states_to_save["out_states"],
                                                        states_to_save["r_gen_states"],
                                                        states_to_save["r_gen_pre_pca_states"],
                                                        n_samples_subset=1000,
                                                        n_ens=100
                                                        )
                std_w_out = np.std(w_out_subsets, axis=0)
                mean_w_out = np.mean(w_out_subsets, axis=0)

            # st.write("Normal w_out")
            # w_out = esn_obj.get_w_out().copy()
            # fig = plpl.matrix_as_barchart(w_out.T)
            # st.plotly_chart(fig)
            #
            # st.write("Std of wout")
            # esnplot.st_plot_w_out_as_barchart(std_w_out)
            # # fig = plpl.matrix_as_barchart(std_w_out.T)
            # # st.plotly_chart(fig)
            #
            # st.write("Mean of wout")
            # fig = plpl.matrix_as_barchart(mean_w_out.T)
            # st.plotly_chart(fig)
            #
            # st.write("Std of wout / mean of w_out")
            # esnplot.st_plot_w_out_as_barchart(std_w_out / mean_w_out, key="std/mean")
            #
            # st.write("Std of wout / mean of w_out up to rank")
            # esnplot.st_plot_w_out_as_barchart((std_w_out / mean_w_out)[:, :rank], key="std/mean rank")

            # # Overwrite esn_w_out with mean_w_out
            # esn_obj._w_out = mean_w_out
            # y_pred, y_pred_true, res_pred_dict, esn_obj = esn.predict_return_res(esn_obj,
            #                                                                      x_pred,
            #                                                                      t_pred_sync)
            # esn_obj = copy.deepcopy(esn_obj)  # needed for the streamlit caching to work correctly.
            # pred_data_dict = {"true": y_pred_true,
            #                   "pred": y_pred}
            # plot_tab, measure_tab, difference_tab = st.tabs(["Plot", "Measures", "Difference"])
            # with plot_tab:
            #     plot.st_all_timeseries_plots(pred_data_dict, key="predict2")
            # with measure_tab:
            #     measures.st_all_data_measures(pred_data_dict, dt=dt, key="predict2")
            # with difference_tab:
            #     pred_vs_true.st_all_difference_measures(y_pred_traj=y_pred,
            #                                             y_true_traj=y_pred_true,
            #                                             dt=dt,
            #                                             train_or_pred="predict",
            #                                             key="predict2")

            if st.checkbox("Cross lyapunov exponent", key="cross lyap exp checkbox"):
                # TODO: Experimental
                st.warning("EXPERIMENTAL")
                st.info("Calculate the \"cross lyapunov exponent\" as a measure for the "
                        "prediction quality. ")

                left, right = st.columns(2)
                with left:
                    steps = int(st.number_input("steps", value=int(1e3),
                                                key=f"cross_lyapunov_exp__steps"))
                with right:
                    part_time_steps = int(st.number_input("time steps of each part", value=15,
                                                          key=f"cross_lyapunov_exp__part"))
                left, right = st.columns(2)
                with left:
                    steps_skip = int(st.number_input("steps to skip", value=50, min_value=0,
                                                     key=f"cross_lyapunov_exp__skip"))
                with right:
                    deviation_scale = 10 ** (
                        float(st.number_input("log (deviation_scale)", value=-10.0,
                                              key=f"cross_lyapunov_exp__eps")))

                iterator_func = syssim.get_iterator_func(system_name, system_parameters)

                lle_conv = rescompmeasures.largest_cross_lyapunov_exponent(
                    iterator_func,
                    y_pred,
                    dt=dt,
                    steps=steps,
                    part_time_steps=part_time_steps,
                    deviation_scale=deviation_scale,
                    steps_skip=steps_skip,
                    return_convergence=True)

                largest_lle = np.round(lle_conv[-1], 5)

                figs = plpl.multiple_1d_time_series({"LLE convergence": lle_conv}, x_label="N",
                                                    y_label="running avg of LLE",
                                                    title=f"Largest Cross Lyapunov "
                                                          f"Exponent: "
                                                          f"{largest_lle}")
                plpl.multiple_figs(figs)
        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')


    #  Container code at the end:
    if build_bool:
        x_dim, r_dim, r_gen_dim, y_dim = esn_obj.get_dimensions()
        with architecture_container:
            esnplot.st_plot_architecture(x_dim=x_dim, r_dim=r_dim, r_gen_dim=r_gen_dim,
                                         y_dim=y_dim)
