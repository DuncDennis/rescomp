"""Streamlit app to predict a timeseries with an Echo State Network.
Author: Dennis Duncan [DuncDennis@gmail.com]"""

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

# TODO: FOR EXPERIMENTAL:
import streamlit_project.app_fragments.esn_experiments as esnexp
import numpy as np
import pandas as pd
import plotly.express as px
import rescomp.data_preprocessing as datapre
import streamlit_project.generalized_plotting.plotly_plots as plpl
import streamlit_project.latex_formulas.esn_pca_formulas as pcalatex
import rescomp.measures_new as rescompmeasures

if __name__ == '__main__':
    st.set_page_config("Ensemble ESN Viewer", page_icon="⚡")

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

        if "dt" in system_parameters.keys():
            dt = system_parameters["dt"]
        else:
            dt = 1.0

        shift_scale_params, noise_scale = syssim.st_preprocess_simulation()

        x_dim_pre = syssim.get_x_dim(system_name, system_parameters)
        embedding_out = syssim.st_embed_timeseries(x_dim_pre, key="embedding")
        embedding_dims, embedding_delay, embedding_dim_selection = embedding_out
        t_pred -= embedding_delay  # embedding makes the timeseries shorter.
        pca_transform_time_series_bool = syssim.st_pca_transform_time_series(key="sidebar")
        utils.st_line()

    with st.sidebar:
        st.header("ESN: ")
        esn_type = esn.st_select_esn_type()
        with st.expander("Basic parameters: "):
            basic_build_args = esn.st_basic_esn_build()
        with st.expander("Network parameters: "):
            build_args = basic_build_args | esn.st_network_build_args()
        if esn_type == "ESN_pca":
            with st.expander("ESN_pca settings: "):
                build_args = basic_build_args | esn.st_pca_build_args(build_args["r_dim"])
        utils.st_line()

    with st.sidebar:
        st.header("Seed: ")
        seed = utils.st_seed()
        utils.st_line()

    sim_data_tab, build_tab, train_tab, predict_tab, other_vis_tab, sweep_tab, theory_tab = \
        st.tabs(
            ["🌀 Simulated data",
             "🛠️ Architecture",
             "🦾 Training",
             "🔮 Prediction",
             "🔬 Look-under-hood",
             "🧹 Sweeping",
             "🔰 Theory"])

    with sim_data_tab:
        if simulate_bool:
            section_steps = [t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred]
            section_names = ["train disc", "train sync", "train", "pred disc", "pred sync", "pred"]
            time_steps = sum(section_steps)

            time_series = syssim.simulate_trajectory(system_name, system_parameters,
                                                     time_steps)

            time_series, scale_shift_vector = syssim.preprocess_simulation(time_series,
                                                                           seed,
                                                                           scale_shift_params=shift_scale_params,
                                                                           noise_scale=noise_scale)

            time_series_dict = {"time series": time_series}

            time_series = syssim.get_embedded_time_series(time_series,
                                                          embedding_dimension=embedding_dims,
                                                          delay=embedding_delay,
                                                          dimension_selection=embedding_dim_selection)
            if pca_transform_time_series_bool:
                time_series = syssim.get_pca_transformed_time_series(time_series)

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
            esn_obj = esn.build(esn_type, seed=seed, x_dim=x_dim, **build_args)
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

            res_states_tab, w_out_r_gen_tab, res_time_tab, res_dyn_tab, pca_tab, more_tab = st.tabs(
                ["Internal reservoir states", "W_out and R_gen",
                 "Reservoir time series", "Pure reservoir dynamics", "PCA esn stuff", "More"])

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

            with res_dyn_tab:
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

            with pca_tab:
                st.info("The following functionalities make sense for PCA_ens where the "
                        "generalized reservoir states represent the PCA components.")
                if st.checkbox("Remove generalized reservoir dimensions and see prediction"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")
                    st.info(
                        "Take the trained PCA-ESN and set all w_out entries that correspond to "
                        "r_gen states that lay between min r_gen_dim and max r_gen_dim to zero.")

                    esn_to_test = copy.deepcopy(esn_obj)
                    r_gen_dim = res_pred_dict["r_gen"].shape[1]
                    cols = st.columns(2)
                    with cols[0]:
                        i_rgen_dim_min = int(
                            st.number_input("min r_gen_dim", value=50, min_value=0,
                                            max_value=r_gen_dim - 1,
                                            key="mod r_gen_dim min"))
                    with cols[1]:

                        i_rgen_dim_max = int(st.number_input("max r_gen_dim", value=r_gen_dim - 1,
                                                             min_value=0,
                                                             max_value=r_gen_dim,
                                                             key="mod r_gen_dim min"))
                    if i_rgen_dim_max == r_gen_dim:
                        esn_to_test._w_out[:, i_rgen_dim_min:] = 0.0
                    else:
                        esn_to_test._w_out[:, i_rgen_dim_min: i_rgen_dim_max] = 0.0

                    y_pred_mod, y_pred_mod_true, _, _ = esn.predict_return_res(
                        esn_to_test,
                        x_pred,
                        t_pred_sync)

                    pred_data_dict = {"true": y_pred_mod_true,
                                      "pred": y_pred_mod}
                    w_out_mod = esn_to_test.get_w_out()

                    esnplot.st_plot_w_out_as_barchart(w_out_mod, key="predict_rmv_r_gen")
                    st.info("Predict with the modified esn and plot the results.")
                    plot.st_all_timeseries_plots(pred_data_dict, key="predict_rmv_r_gen")

                utils.st_line()
                if st.checkbox(
                        "Correlate generalized reservoir states with input during training"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")
                    st.info("Take the input used for training (x_train[t_train_sync: -1, :])"
                            "and calculate the dimension wise correlation with the driven "
                            "generalized reservoir states, i.e. correlate every input dimension "
                            "with every r_gen dimension. The time delay shifts the input "
                            "dimension in the past, i.e. correlate r_gen[time_delay:] "
                            "with inp[: - time_delay. "
                            "For negative time_delays correlate r_gen with future inp states.")

                    inp = x_train[t_train_sync:-1, :]
                    r_gen = res_train_dict["r_gen"]

                    time_delay = int(st.number_input("time_delay", value=0))

                    correlation = esnexp.correlate_input_and_r_gen(inp, r_gen, time_delay)

                    l, m, r = st.columns(3)
                    with l:
                        log_y = st.checkbox("log_y", key="log_123")
                    with m:
                        barmode = st.selectbox("barmode", ["relative", "group", "subplot"])
                    with r:
                        abs_bool = st.checkbox("abs", value=True)
                    title = "correlation r_gen vs input"
                    x_axis = "r_gen_dim"
                    y_axis = "input_dim"
                    value_name = "correlation"
                    st.info("Plot correlation as a bar plot:")
                    fig = plpl.matrix_as_barchart(correlation,
                                                  log_y=log_y,
                                                  barmode=barmode,
                                                  abs_bool=abs_bool,
                                                  title=title,
                                                  x_axis=x_axis, y_axis=y_axis,
                                                  value_name=value_name)
                    st.plotly_chart(fig)
                utils.st_line()

                if st.checkbox(
                        "Time delay sweep: Correlate generalized reservoir states with input "
                        "during training"):
                    # TODO: experimental
                    st.warning("EXPERIMENTAL")
                    st.info(
                        "Calculate the same correlation as in \"Correlate generalized reservoir "
                        "states with input during training\". "
                        "The time delay is sweeped on the x axis. For every time_delay "
                        "calculate for every input dimension the \"mean r_gen dimension\", "
                        "i.e. the \"center of gravity\" of the correlation barplot above. ")
                    st.latex(
                        r"\tilde{c}_{i, j} = |c_{i, j}| / \sum_i^{\text{r gen dim}} |c_{i, j}|, \qquad i: \text{r gen dim}, j: \text{input dim}")
                    st.latex(
                        r"\text{mean r gen dimension} = \sum_{i=1}^\text{r gen dim} \tilde{c}_{i, j} \times i")

                    inp = x_train[t_train_sync:-1, :]
                    r_gen = res_train_dict["r_gen"]
                    r_gen_dim_temp = r_gen.shape[1]

                    cols = st.columns(2)
                    with cols[0]:
                        min_time_delay = int(st.number_input("min time delay", value=-2))
                    with cols[1]:
                        max_time_delay = int(st.number_input("max time delay", value=10))
                    time_delays = np.arange(start=min_time_delay, stop=max_time_delay, dtype=int)

                    results = np.zeros((time_delays.size, x_dim))

                    for i_t, time_delay in enumerate(time_delays):
                        correlation = esnexp.correlate_input_and_r_gen(inp, r_gen, time_delay)
                        correlation[np.isnan(correlation)] = 0
                        correlation = np.abs(correlation)
                        correlation = correlation / np.sum(correlation, axis=0)
                        correlation_times_r_gen_index = (correlation.T * np.arange(1,
                                                                                   r_gen_dim_temp + 1)).T

                        correlation_sum = np.sum(correlation_times_r_gen_index, axis=0)
                        # total_sum = np.sum(correlation_sum)
                        results[i_t, :] = correlation_sum

                    fig = px.line(results)
                    fig.update_yaxes(title="mean r gen dimension")
                    fig.update_xaxes(title="time delay")
                    fig.update_layout(
                        xaxis=dict(
                            tickmode='array',
                            tickvals=np.arange(time_delays.size),
                            ticktext=[str(x) for x in time_delays],
                        )
                    )
                    st.plotly_chart(fig)
                    st.info("The hope is, that the \"memory of the reservoir\" is saved in the "
                            "higher pca components. As the \"mean r_gen dim\" increases with the "
                            "time_delay, this might be correct. On the other hand it be another"
                            "unaccounted effect. ")

                utils.st_line()
                if st.checkbox("Remove network and see differences"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")
                    st.info("Take the same esn as used for the training and remove the internal "
                            "reservoir update function, i.e. set the network to 0. "
                            "Now train the modified reservoir again on the same data and see how "
                            "the pca components change. This tests the conjecture, that the first, "
                            "most important, pca states correspond mostly to input, and the higher "
                            "ones are responsible for the \"memory\". ")
                    esn_to_test = copy.deepcopy(esn_obj)
                    esn_to_test._res_internal_update_fct = lambda r: 0

                    y_train_mod_fit, y_train_mod_true, res_train_mod_dict, esn_to_test = esn.train_return_res(
                        esn_to_test,
                        x_train,
                        t_train_sync,
                    )
                    esn_to_test = copy.deepcopy(esn_to_test)
                    w_out_no_network = esn_to_test.get_w_out()

                    r_gen_mod_dict = {"r_gen with network": res_train_dict["r_gen"],
                                      "r_gen without network": res_train_mod_dict["r_gen"]}

                    st.info(
                        "Plot the r_gen time series during training with and without the network: ")
                    plot.st_plot_dim_selection(r_gen_mod_dict, key="rmv_network")

                    st.info("Plot the different w_outs as barcharts: ")
                    esnplot.st_plot_w_out_as_barchart(w_out, key="predict_rmv_network1")
                    esnplot.st_plot_w_out_as_barchart(w_out_no_network, key="predict_rmv_network2")

                    st.info("Calculate for each r gen dimension, the error between the true and "
                            "the modified r_gen. ")
                    st.latex(
                        r"\text{error over time} = (\tilde{r}_i(t) - r_i(t))^2 / \sqrt{<r_i^2>_t}")
                    st.latex(r"\tilde{r_i}: \text{gen res state i without network}")
                    st.latex(r"r_i: \text{gen res state i with network}")
                    st.latex(
                        r"\text{Total error per r gen dim} = \sum_\text{time} \text{error over time}")

                    st.info(
                        "Beware that sometimes the important pca states are actually nearly the same but flipped,"
                        "resulting in a big error.")

                    r_gen_difference = res_train_dict["r_gen"] - res_train_mod_dict["r_gen"]
                    error_over_time = np.abs(r_gen_difference) / np.sqrt(
                        np.mean(np.square(res_train_dict["r_gen"]), axis=0))

                    error_over_r_dim = np.sum(error_over_time, axis=0)
                    fig = px.line(error_over_r_dim)
                    fig.update_xaxes(title="r gen index")
                    fig.update_yaxes(title="Total error per r gen dim")
                    st.plotly_chart(fig)

                    st.info("We can see that the first components indeed are very similar, i.e. "
                            "have a small error. ")

                    st.info("Also predict with the modified esn and compare: ")
                    y_pred_mod, y_pred_mod_true, _, _ = esn.predict_return_res(
                        esn_to_test,
                        x_pred,
                        t_pred_sync)

                    pred_data_dict = {"true": y_pred_mod_true,
                                      "pred with network": y_pred,
                                      "pred without network": y_pred_mod,
                                      }
                    plot.st_all_timeseries_plots(pred_data_dict, key="predict_rmv network")

                utils.st_line()
                if st.checkbox("Drive trained reservoir with fading signal"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")
                    st.info("Drive a trained (pca) esn with a signal and gradually / fast turn of "
                            "the signal and see the response in the generalized reservoir states. "
                            "Generally the reservoir state echo falls off really fast. "
                            "The idea is, that the higher pca states fade out slower. ")
                    esn_to_test = copy.deepcopy(esn_obj)

                    x_pred_fade_out = copy.deepcopy(x_pred)
                    x_pred_time_steps, x_dim = x_pred.shape

                    st.info("Specify and plot the fading out signal: ")

                    cols = st.columns(2)
                    with cols[0]:
                        fade_out_start = int(st.number_input("Fade out start", value=300))
                    with cols[1]:
                        fade_out_mode = st.selectbox("fade out mode", ["instant", "exponential"])

                    if fade_out_mode == "instant":
                        x_pred_fade_out[fade_out_start:, :] = np.zeros(x_dim)
                    elif fade_out_mode == "exponential":
                        factor = 0.5
                        x_pred_fade_out[fade_out_start:, :] = x_pred_fade_out[fade_out_start:, :] * \
                                                              np.repeat(np.exp(-factor * np.arange(
                                                                  x_pred_time_steps - fade_out_start))[
                                                                        :, np.newaxis],
                                                                        x_dim, axis=1
                                                                        )
                    to_plot = {"true": x_pred, "faded out": x_pred_fade_out}
                    plot.st_plot_dim_selection(to_plot, key="fading")

                    st.info("Drive the reservoir with the fading out signal and plot a response. "
                            "The r_gen states are rescaled by their std, for better"
                            " vizualization. ")

                    r_gen_fadeout, _ = esnexp.drive_reservoir(esn_to_test, x_pred_fade_out)
                    r_gen_fadeout_scaled = r_gen_fadeout / np.std(r_gen_fadeout, axis=0)
                    r_gen_fadeout_scaled[np.isinf(r_gen_fadeout_scaled)] = 0
                    r_gen_dim = r_gen_fadeout.shape[1]
                    to_plot = {"r_gen fadeout rescaled": r_gen_fadeout_scaled,
                               "x_pred fadeout (axis=0)": np.repeat(x_pred_fade_out[:, 0:1],
                                                                    r_gen_dim, axis=1)}

                    plot.st_plot_dim_selection(to_plot, key="r_gen fade")

                    st.info("Calculate the difference of the scaled r_gen, i.e. r[1:] - r[:-1]")
                    diff = np.diff(r_gen_fadeout_scaled, axis=0)
                    to_plot = {"diff": diff}
                    plot.st_plot_dim_selection(to_plot, key="r_gen fade diff")

                    st.info(
                        "For each generalized reservoir dimension, starting from the \"fade out start\", "
                        "calculate the min-max-spread in the difference for a sliding window. "
                        "If the min-max-spread is smaller than a threshold, the reservoir is considered from "
                        "this index on to be constant. The time index is saved. "
                        "I.e. one saves the earliest time index, where the reservoirs echo is "
                        "killed off. ")
                    out = diff[fade_out_start:, :]
                    # out = r_gen_fadeout_scaled[fade_out_start:, :]

                    cols = st.columns(3)
                    with cols[0]:
                        time_steps_to_test = int(
                            st.number_input("time steps to test", value=40, key="testimesteps"))
                    with cols[1]:
                        pnts_to_try = int(
                            st.number_input("points in sliding window", value=5, key="pntswindow"))
                    with cols[2]:
                        threshhold = st.number_input("threshhold", value=0.00001, key="threshtest",
                                                     step=0.001, format="%f")

                    index_to_save = np.zeros(r_gen_dim)
                    for i_r_dim in range(r_gen_dim):
                        min_index_set = False
                        for i in range(time_steps_to_test):  # time steps.
                            slice = out[i: i + pnts_to_try, i_r_dim]
                            min_max = np.ptp(slice)  # for each slice the
                            if min_max < threshhold:
                                min_index_set = True
                                index_to_save[i_r_dim] = i  # + fade_out_start
                                break

                        if not min_index_set:
                            index_to_save[i_r_dim] = time_steps_to_test

                    fig = px.line(index_to_save)
                    fig.update_yaxes(title="time_step after fade_out_start")
                    fig.update_xaxes(title="r gen dimension")
                    fig.update_layout(title="Echo in r gen dimensions")
                    st.plotly_chart(fig)
                    st.info(
                        "One can slightly see that the first pca components have on average a lower memory.")

                utils.st_line()
                if st.checkbox("Surrogate input as comparison"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")
                    st.info(
                        "Create the Fourier transform surrogates of the original time series (actually of x_train), "
                        "and drive the trained reservoir with it and see the r_gen and w_out distribution. ")

                    st.info(
                        "Select the seed used for the surrogates, the seeds for each dimension "
                        "will be seed + i_x_dim:")
                    seed = int(
                        st.number_input("seed for surrogates", value=0, key="surrogateseed"))

                    surrogate_time_series = datapre.fourier_transform_surrogate(x_train, seed=seed)

                    st.info(
                        "Plot the power spectrum for the surrogate and real x_train time series. "
                        "It will be the same, by definition. ")
                    to_plot = {"surrogate": surrogate_time_series, "real": x_train}
                    measures.st_power_spectrum(to_plot, key="surrogatepower")

                    st.info("Plot the surrogate time series vs. the real time series. "
                            "The surrogate time_series lost the structure.")

                    plot.st_plot_dim_selection(to_plot)

                    st.info("Drive the reservoir")

                    esn_to_test = copy.deepcopy(esn_obj)

                    _, _, res_states_surrogate, esn_to_test = esn.train_return_res(esn_obj,
                                                                                   surrogate_time_series,
                                                                                   t_train_sync)
                    w_out_surrogate = esn_to_test.get_w_out()
                    st.info("Plot the different w_outs as barcharts: ")
                    esnplot.st_plot_w_out_as_barchart(w_out, key="surrogate normal wout")
                    esnplot.st_plot_w_out_as_barchart(w_out_surrogate,
                                                      key="surrogate surrogate wout")

                    st.info("Show the std of r_gen states for the surrogate and the normal train. "
                            "Confusing labels: r_gen_train = real r_gen, r_gen_pred: surrogate r_gen. ")
                    esnplot.st_r_gen_std_barplot(res_train_dict["r_gen"],
                                                 res_states_surrogate["r_gen"])

                utils.st_line()
                if st.checkbox("Analyize which input dimensions go into the pca components"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")

                    if hasattr(esn_obj, "_pca"):
                        pca_components = esn_obj._pca.components_  # shape: [n_components, n_features]
                    else:
                        raise Exception("ESN object does not have _pca. ")

                    st.info("Idea: It seems like the first pca components are dominated by the "
                            "input coupling in the reservoir nodes. Here we try to quantify how "
                            "much of the individual input dimensions go into the pca components. ")

                    st.info("The PCA transformation is substraction by a constant vector, and the "
                            "multiplication of a unitary matrix: ")
                    st.latex(pcalatex.pca_transformation)
                    st.latex(pcalatex.pca_transformation_definition)
                    st.info("The reservoir state in its own coordinate system can be written as: ")
                    st.latex(pcalatex.reservoir_states)
                    st.latex(pcalatex.res_state_unit_vectors_explanation)

                    st.info("The inverse pca transformation is given as: ")
                    st.latex(pcalatex.pca_inverse_transformation)

                    st.info(
                        "Then the pca components in the resrevoir state coordinate system are given as: ")
                    st.latex(pcalatex.pca_components)

                    if st.checkbox("Show pca components as an image: ", key="pca_comp_as_image"):
                        fig = px.imshow(pca_components)
                        fig.update_xaxes(title="reservoir nodes")
                        fig.update_yaxes(title="pca component")
                        st.plotly_chart(fig)
                    if st.checkbox("Show pca components as line plots: ",
                                   key="pca_comp_as_lineplot"):
                        fig = px.line(pca_components)
                        fig.update_xaxes(title="reservoir index")
                        fig.update_yaxes(title="m_j")
                        fig.for_each_trace(lambda t: t.update(name=f"PC: {t.name}",
                                                              legendgroup=f"PC: {t.name}",
                                                              hovertemplate=t.hovertemplate.replace(
                                                                  t.name, f"PC: {t.name}")
                                                              )
                                           )
                        st.plotly_chart(fig)
                    st.info(
                        "If we approximate the reservoir states just as a function of the input: ")
                    st.latex(pcalatex.reservoir_as_function_of_input_approx)

                    st.info("We can write the pca transformed vector approximately as: ")
                    st.latex(pcalatex.pca_vector_as_fct_of_input)

                    st.info("If the input vector is given as: ")
                    st.latex(pcalatex.input_states)

                    st.info("The resulting modified pca component vectors are given as:")
                    st.latex(pcalatex.pca_components_as_fct_of_input)

                    st.info("Now we can estimate how much of each input dimension goes into a "
                            "pca components. For a three dimensional input the modified pca "
                            "component would be a three dimensional vector: ")

                    st.latex(r"\boldsymbol{n}_j = [n_{j, 1}, n_{j, 2}, n_{j, 3}]^\intercal")

                    w_in = esn_obj._w_in
                    if st.checkbox(
                            "Show absolute values of n_j as a matrix barplot, compare with w_out, "
                            "and input-r_gen correlation (timedelay=0).",
                            key="w_in_pca_n_j_barplot"):

                        # get n:
                        n = pca_components @ w_in

                        # get w_out:
                        w_out = esn_obj.get_w_out()

                        # get correlation:
                        inp = x_train[t_train_sync:-1, :]
                        r_gen = res_train_dict["r_gen"]
                        correlation = esnexp.correlate_input_and_r_gen(inp, r_gen, time_delay=0)

                        max_index = int(st.number_input("max index to show: ", value=5, step=1,
                                                        key="w_in_pca_n_j_barplot__max_index"))

                        abs_bool = st.checkbox("Absolute values", value=True,
                                               key="w_in_pca_n_j_barplot__absbool")

                        normalize_by_sum = st.checkbox("Normalize matrix by sum",
                                                       key="w_in_pca_n_j_barplot__normalize",
                                                       disabled=not abs_bool)

                        log_y = st.checkbox("log y", key=f"w_in_pca_n_j_barplot__logy",
                                                       disabled=not abs_bool)

                        if normalize_by_sum:
                            n = (np.abs(n).T / np.sum(np.abs(n), axis=1)).T
                            w_out = (np.abs(w_out) / np.sum(np.abs(w_out), axis=0))
                            correlation = (np.abs(correlation).T / np.sum(np.abs(correlation),
                                                                          axis=1)).T

                        st.markdown("**n plot:**")

                        fig = plpl.matrix_as_barchart(n[:max_index, :], x_axis="pca component",
                                                      y_axis="input dim",
                                                      value_name="n", log_y=log_y, abs_bool=abs_bool)
                        st.plotly_chart(fig)

                        st.markdown("**w_out plot:**")

                        fig = plpl.matrix_as_barchart(w_out[:, :max_index].T, x_axis="r_gen index",
                                                      y_axis="out dim",
                                                      value_name="w_out", log_y=log_y,
                                                      abs_bool=abs_bool)
                        st.plotly_chart(fig)

                        st.markdown("**Correlation:**")

                        fig = plpl.matrix_as_barchart(correlation[:max_index, :], x_axis="r_gen index",
                                                      y_axis="input dim",
                                                      value_name="correlation", log_y=log_y,
                                                      abs_bool=abs_bool)
                        st.plotly_chart(fig)
                        st.info("We can see clearly that the first pca nodes get the input dimensions "
                                "in the same relations as w_out fitting the output dimensions. ")

                utils.st_line()
                if st.checkbox("Analyize which reservoir nodes go into the pca components"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")

                    if hasattr(esn_obj, "_pca"):
                        pca_components = esn_obj._pca.components_  # shape: [n_components, n_features]
                    else:
                        raise Exception("ESN object does not have _pca. ")
                    st.info("Show pca components as an image: ")
                    st.latex(r"\text{pca} = [a_1, a_2, ...] = \sum_i a_i \boldsymbol{r}_i")
                    fig = px.imshow(pca_components)
                    fig.update_xaxes(title="reservoir nodes")
                    fig.update_yaxes(title="pca component")
                    st.plotly_chart(fig)
                    st.info(
                        "Calculate and plot a percentaged pca component image and individual components:")

                    st.latex(
                        r"\text{pca percentage of reservoir nodes} = [|a_1|, |a_2|, ...] / \sum|a_i|"
                        r"= [c_1, c_2, ...]")
                    pca_components_percentage = (
                                np.abs(pca_components).T / np.sum(np.abs(pca_components),
                                                                  axis=1)).T
                    fig = px.imshow(pca_components_percentage)
                    fig.update_xaxes(title="reservoir nodes")
                    fig.update_yaxes(title="pca component percentage")
                    st.plotly_chart(fig)

                    if st.checkbox("Plot pca component percentage vs. reservoir index: ",
                                   key="pcaperc_vs_res"):
                        fig = px.line(pca_components_percentage.T)
                        fig.update_xaxes(title="reservoir index")
                        fig.update_yaxes(title="pca component percentage")
                        st.plotly_chart(fig)

                utils.st_line()
                if st.checkbox("Split the output of the generalized reservoir nodes and see driving "
                               "with x_train: ", key="splitoutput_rgen"):
                    # TODO: EXPERIMENTAL
                    st.warning("EXPERIMENTAL")
                    st.info(
                        "Take the esn r_gen states used for training and split the states at some "
                        "r_gen dimensions. For each side of the split (The first and the last r_gen "
                        "states) calculate the partial output that they produce. ")

                    esn_to_test = copy.deepcopy(esn_obj)
                    r_gen_states = res_train_dict["r_gen"]
                    r_gen_dim = r_gen_states.shape[1]
                    w_out = esn_to_test.get_w_out()

                    i_rgen_dim_split = int(
                        st.number_input("split r_gen_dim", value=50, min_value=0,
                                        max_value=r_gen_dim - 1,
                                        key="r_gen_dim split"))

                    r_gen_first = r_gen_states[:, :i_rgen_dim_split]
                    r_gen_last = r_gen_states[:, i_rgen_dim_split:]

                    esn_output_first = (w_out[:, :i_rgen_dim_split] @ r_gen_first.T).T
                    esn_output_last = (w_out[:, i_rgen_dim_split:] @ r_gen_last.T).T
                    to_corr_y_dict = {"esn output first": esn_output_first,
                                      "esn output last": esn_output_last}

                    st.write("nr of first r_gen dims", r_gen_first.shape[1],
                             "nr of last r_gen dims", r_gen_last.shape[1])
                    plot.st_default_simulation_plot_dict(to_corr_y_dict)

                    st.info("As a comparison plot the real input (the data the esn was driven with), "
                            "The real output (what the r_gen states were fitted to)"
                            "and the difference to the next time step. ")

                    inp = x_train[t_train_sync:-1, :]
                    out = x_train[t_train_sync + 1:, :]
                    to_corr_x_dict = {"real input": inp,
                                      "real output": out,
                                      "real difference": out - inp}
                    plot.st_default_simulation_plot_dict(to_corr_x_dict)

                    st.info("Now compare the first esn output with the real input and real output: ")
                    to_plot = {"esn_output_first": esn_output_first,
                               "real input": inp,
                               "real output": out}

                    plot.st_default_simulation_plot_dict(to_plot)
                    plot.st_plot_dim_selection(to_plot, key="output split input")

                    st.info("Now compare the esn output last with the real difference: ")
                    to_plot = {"esn_output_last": esn_output_last, "real difference": out - inp}
                    plot.st_default_simulation_plot_dict(to_plot)
                    plot.st_plot_dim_selection(to_plot, key="output split")

                    st.info("Now correlate both partial esn outputs with the real input and the "
                            "real difference. The correlation value is the sum of the diagonal "
                            "elements of the correlation matrix. "
                            "I.e. Only a dimension is correlated with the same dimension.")

                    corr_data_summed_dict = {  # type: ignore
                        "correlation x": [],
                        "correlation y": [],
                        "correlation value": []
                    }

                    for x_name, x_data in to_corr_x_dict.items():
                        for y_name, y_data in to_corr_y_dict.items():
                            corr_multidim = esnexp.correlate_input_and_r_gen(x_data,
                                                                             y_data,
                                                                             time_delay=0)
                            # corr_summed = np.sum(corr_multidim)
                            corr_summed = np.sum(np.diag(corr_multidim))
                            # corr_summed = np.sum(np.abs(corr_multidim))
                            corr_data_summed_dict["correlation x"].append(x_name)
                            corr_data_summed_dict["correlation y"].append(y_name)
                            corr_data_summed_dict["correlation value"].append(corr_summed)
                    fig = px.bar(corr_data_summed_dict, x="correlation x", y="correlation value",
                                 color="correlation y", barmode="group")
                    st.plotly_chart(fig)

                    with st.expander("Findings: "):
                        st.info("When using a linear activation function: "
                                "The dimensionwise first_esn_output vs. input vs. output plot "
                                "shows, that: "
                                "For linear dimensions (i.e. linear term in the system "
                                "equation) the first_esn_output plot is synchronized with the output. "
                                "probably because its easier to predict. "
                                "For the non-linear dimensions (i.e. non-linear terms in the "
                                "system equations) first_esn_output is synchronized with the input "
                                "(probably since the other nodes are needed to fit the output. ")

                    st.write(r_gen_states.shape, x_train.shape, inp.shape)

                    # Dimension wise correlation (no cross correlation between dimension).
                    if st.checkbox("Dimension wise correlation: ",
                                   key="dimwise correlation split rgen"):
                        corr_data_dict = {"correlation x": [], "correlation y": [],
                                          "correlation value": [], "dimension": []}
                        for x_name, x_data in to_corr_x_dict.items():
                            for y_name, y_data in to_corr_y_dict.items():
                                corr_multidim = esnexp.correlate_input_and_r_gen(x_data,
                                                                                 y_data,
                                                                                 time_delay=0)

                                for i_dim in range(corr_multidim.shape[0]):
                                    corr_value = corr_multidim[i_dim, i_dim]
                                    corr_data_dict["dimension"].append(i_dim)
                                    corr_data_dict["correlation x"].append(x_name)
                                    corr_data_dict["correlation y"].append(y_name)
                                    corr_data_dict["correlation value"].append(corr_value)
                        df_corr = pd.DataFrame.from_dict(corr_data_dict)

                        fig = px.bar(df_corr, x="correlation x",
                                     y="correlation value",
                                     color="correlation y",
                                     barmode="group",
                                     facet_row="dimension")
                        st.plotly_chart(fig)
                        st.write(df_corr)

                    if st.checkbox("Dimension wise correlation (also cross dimensions): ",
                                   key="dimwise cross-correlation split rgen"):
                        # Multidimensional correlation:
                        corr_data_dict = {"correlation x": [], "correlation y": [],
                                          "correlation value": [], "x_dim": [], "y_dim": []}
                        for x_name, x_data in to_corr_x_dict.items():
                            for y_name, y_data in to_corr_y_dict.items():
                                corr_multidim = esnexp.correlate_input_and_r_gen(x_data,
                                                                                 y_data,
                                                                                 time_delay=0)

                                for i_x in range(corr_multidim.shape[0]):
                                    for i_y in range(corr_multidim.shape[1]):
                                        corr_value = corr_multidim[i_x, i_y]
                                        corr_data_dict["x_dim"].append(i_x)
                                        corr_data_dict["y_dim"].append(i_y)
                                        corr_data_dict["correlation x"].append(x_name)
                                        corr_data_dict["correlation y"].append(y_name)
                                        corr_data_dict["correlation value"].append(corr_value)
                        df_corr = pd.DataFrame.from_dict(corr_data_dict)
                        fig = px.bar(df_corr, x="correlation x",
                                     y="correlation value",
                                     color="correlation y",
                                     barmode="group",
                                     facet_row="x_dim",
                                     facet_col="y_dim")
                        st.plotly_chart(fig)
                        st.write(df_corr)
                utils.st_line()
                if st.checkbox("W_out pca vs W_out normal ESN: ", key="w_out_pca vs_ w_out normal"):
                    # TODO: Experimental
                    st.warning("EXPERIMENTAL")
                    st.info("This experiment assumes that the specified esn on the right is a ESN_pca.")
                    if esn_type != "ESN_pca":
                        st.warning("ESN type must be \"ESN_pca\"")

                    esn_to_test = copy.deepcopy(esn_obj)
                    if hasattr(esn_to_test, "_pca"):
                        pca_components = esn_to_test._pca.components_  # shape: [n_components, n_features]
                    else:
                        raise Exception("ESN object does not have _pca. ")

                    r_dim = build_args["r_dim"]
                    r_gen_dim = res_train_dict["r_gen"].shape[1]

                    w_out_pca = esn_to_test.get_w_out()
                    if r_dim < r_gen_dim:
                        w_out_pca = w_out_pca[:, :r_dim]

                    esn_type_comparison = st.selectbox("ESN normal or normal centered: ", ["ESN_normal", "ESN_normal_centered"])

                    esn_normal = esn.build(esn_type=esn_type_comparison, seed=seed, x_dim=x_dim,
                                           **build_args)

                    esn_normal = copy.deepcopy(esn_normal)
                    y_train_fit_normal, y_train_true_normal, res_train_dict_normal, esn_normal = \
                        esn.train_return_res(
                            esn_normal,
                            x_train,
                            t_train_sync,
                        )

                    w_out_normal = esn_normal.get_w_out()
                    if r_dim < r_gen_dim:
                        w_out_normal = w_out_normal[:, :r_dim]

                    w_out_pca_times_pca = w_out_pca @ pca_components

                    w_out_pca_from_normal = w_out_normal @ np.linalg.inv(pca_components)

                    st.info("Both as PCA w_out (first real pca, second from normal and pca_components):")
                    st.markdown("**w_out_pca:**")
                    esnplot.st_plot_w_out_as_barchart(w_out_pca, key="pca original")
                    st.markdown("**w_out_normal @ pca_components:**")
                    esnplot.st_plot_w_out_as_barchart(w_out_pca_from_normal, key="pca from normal")
                    st.markdown("**Difference:**")
                    esnplot.st_plot_w_out_as_barchart(w_out_pca - w_out_pca_from_normal,
                                                      key="pca vs pca from normal difference")

                    st.info("Both as normal w_out (first real normal, second from pca and pca_components):")
                    st.markdown("**w_out_normal**")
                    esnplot.st_plot_w_out_as_barchart(w_out_normal, key="normal original")
                    st.markdown("**w_out_pca @ pca_components ^(-1)**")
                    esnplot.st_plot_w_out_as_barchart(w_out_pca_times_pca, key="normal from pca")
                    st.markdown("**Difference:**")
                    esnplot.st_plot_w_out_as_barchart(w_out_normal - w_out_pca_times_pca,
                                                      key="normal vs pca normal difference")

                    st.write("ESN offset: ", esn_to_test._input_data_mean)

            with more_tab:
                if st.checkbox("Cross lyapunov exponent", key="cross lyap exp checkbox"):
                    # TODO: Experimental
                    st.warning("EXPERIMENTAL")
                    st.info("Calculate the \"cross lyapunov exponent\" as a measure for the "
                            "prediction quality. ")

                    iterator_func = syssim.get_iterator_func(system_name, system_parameters)
                    sysmeas.st_largest_cross_lyapunov_exponent(iterator_func,
                                                               y_pred,
                                                               scale_shift_vector=scale_shift_vector,
                                                               dt=dt,
                                                               )

        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')

    with sweep_tab:
        if predict_bool:
            tabs = st.tabs(["PCA on reservoir states"])
            with tabs[0]:  # PCA on reservoir states
                if st.checkbox("Explained var of PCA comps for sweep",
                               key="sweep__pca_expl_var"):
                    from sklearn.decomposition import PCA

                    N_ens = 3
                    r_or_r_gen = "r"  # or "r_gen"

                    sweep_name = "n_rad"
                    # sweep_vals = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
                    sweep_vals = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.9]

                    # sweep_vals = ["linear_r", "linear_and_square_r"]
                    # sweep_name = "r_to_r_gen_opt"

                    # sweep_name = "w_in_scale"
                    # sweep_vals = [0.01, 0.1, 1, 10, 100, 1000]

                    # sweep_vals = ["tanh", "sigmoid", "relu", "linear"]
                    # # # # sweep_vals = ["tanh", "sigmoid"]
                    # sweep_name = "act_fct_opt"

                    results_dict = {sweep_name: [],  # type: ignore
                                    "explained variance": [],
                                    # "var": [],
                                    # "std": [],
                                    "pca index": [],
                                    "ensemble": []}

                    for sweep_value in sweep_vals:
                        sweep_build_args = copy.deepcopy(build_args)
                        sweep_build_args[sweep_name] = sweep_value
                        for i_ens in range(N_ens):
                            sweep_seed = seed + i_ens
                            sweep_esn = esn.build(esn_type=esn_type,
                                                  seed=sweep_seed,
                                                  x_dim=x_dim,
                                                  **sweep_build_args)
                            sweep_esn = copy.deepcopy(sweep_esn)
                            _, _, res_train_dict_sweep, _ = esn.train_return_res(
                                sweep_esn,
                                x_train,
                                t_train_sync,
                            )
                            res_states_sweep = res_train_dict_sweep[r_or_r_gen]

                            pca = PCA()
                            res_states_pca = pca.fit_transform(res_states_sweep)
                            res_states_pca_dims = res_states_pca.shape[1]

                            results_dict["pca index"] += np.arange(res_states_pca_dims).tolist()

                            # var = np.var(res_states_pca, axis=0)
                            # results_dict["var"] += var.tolist()

                            results_dict[sweep_name] += [sweep_value, ] * res_states_pca_dims

                            results_dict["ensemble"] += [i_ens, ] * res_states_pca_dims
                            results_dict["explained variance"] += pca.explained_variance_.tolist()
                    df = pd.DataFrame.from_dict(results_dict)

                    fig = px.line(df,
                                  x="pca index",
                                  y="explained variance",
                                  color=sweep_name,
                                  line_group="ensemble",
                                  log_y=True)
                    fig.update_yaxes(type="log", exponentformat="E")
                    st.plotly_chart(fig)
                    st.write(df)
        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')

    with theory_tab:
        if predict_bool:
            if st.checkbox("Check linear ESN without network: ", key="theory_linear_nonet"):
                st.markdown("**Theory:**")
                st.markdown("Here we assume a linear activation function, and no network. "
                            "Furthermore we assume a output-bias readout and an optional node bias. "
                            "With these assumptions the output becomes a function of the input: ")

                st.latex(r"""
                \begin{aligned}
                \boldsymbol{y} &= \tilde{W}_\text{out} \tilde{\boldsymbol{r}} =  W_\text{out} 
                \boldsymbol{r} + \boldsymbol{w} = 
                W_\text{out}\left[ W_\text{in} 
                \boldsymbol{x} + \boldsymbol{b} \right] + \boldsymbol{w} \\ 
                \boldsymbol{y} &= 
                W_\text{out} W_\text{in}  \boldsymbol{x} + W_\text{out} \boldsymbol{b} + \boldsymbol{w}
                \end{aligned}
                """)

                st.markdown("For the pca esn the output looks like: ")
                st.latex(r"""
                \begin{aligned}
                \boldsymbol{y} &= 
                \tilde{W}^\text{pca}_\text{out} \tilde{\boldsymbol{r}}_\text{pca} = 
                W_\text{out}^\text{pca} \boldsymbol{r}_\text{pca} + \boldsymbol{w}_\text{pca} = 
                W_\text{out}^\text{pca} P (\boldsymbol{r} - \boldsymbol{r}_\text{mean}) + 
                \boldsymbol{w}_\text{pca} \\
                
                \boldsymbol{y} &= 
                W_\text{out}^\text{pca} P W_\text{in}(\boldsymbol{x} - \boldsymbol{x}_\text{mean}) + 
                \boldsymbol{w}_\text{pca} \\
                
                \boldsymbol{y} &= 
                W_\text{out}^\text{pca} P W_\text{in}  \boldsymbol{x} +  \boldsymbol{w}_\text{pca} - 
                W_\text{out}^\text{pca} P W_\text{in} \boldsymbol{x}_\text{mean}
                
                \end{aligned}
                """)

                st.markdown("By comparing both equations we get the following relationships: ")
                st.latex(r"""
                \begin{aligned}
                W_\text{out}^\text{pca} P &= W_\text{out} \\
                \boldsymbol{w}_\text{pca} - 
                W_\text{out}^\text{pca} P W_\text{in} \boldsymbol{x}_\text{mean} &=  
                W_\text{out} \boldsymbol{b} + \boldsymbol{w}
                \end{aligned}
                """)

                st.markdown("These two relationships will now be tested numerically:")
                st.info("Please select r_to_r_gen_opt = output_bias, act_fct_opt=linear, node_bias_opt=random_bias,"
                        " n_rad=0.0")
                # normal esn:
                normal_esn = esn.build(esn_type="ESN_normal",
                                       seed=seed,
                                       x_dim=x_dim,
                                       **build_args)
                r_dim = build_args["r_dim"]
                normal_esn = copy.deepcopy(normal_esn)
                _, _, _, normal_esn = esn.train_return_res(
                    normal_esn,
                    x_train,
                    t_train_sync,
                )
                w_out_normal = normal_esn.get_w_out()
                w_out_res_normal = w_out_normal[:, :r_dim]
                w_out_vec_normal = w_out_normal[:, -1]
                b_normal = normal_esn._node_bias

                # PCA esn:
                pca_esn = esn.build(esn_type="ESN_pca",
                                       seed=seed,
                                       x_dim=x_dim,
                                       **build_args)
                pca_esn = copy.deepcopy(pca_esn)
                _, _, _, pca_esn = esn.train_return_res(
                    pca_esn,
                    x_train,
                    t_train_sync,
                )
                w_out_pca = pca_esn.get_w_out()
                w_out_res_pca = w_out_pca[:, :r_dim]
                w_out_vec_pca = w_out_pca[:, -1]
                x_mean = np.mean(x_train[t_train_sync: -1, :], axis=0)
                pca_components = pca_esn._pca.components_
                w_in = pca_esn._w_in

                # first test:  w_out matrices
                pca_matrix = w_out_res_pca @ pca_components
                normal_matrix = w_out_res_normal

                st.write("Difference of matrices: ", pca_matrix - normal_matrix)

                # st.write("W_in as comparison: ", w_in.T)
                # st.write("normal w_out matrix: ", normal_matrix)
                # st.write("pca w_out matrix: ", w_out_res_pca)

                # second test:
                pca_vector = w_out_vec_pca - w_out_res_pca @ pca_components @ w_in @ x_mean
                normal_vector = w_out_res_normal @ b_normal + w_out_vec_normal
                st.write("Differences of vectors: ", pca_vector - normal_vector)

                st.info("For small regulation parameters the difference in matrices gets bigger, "
                        "due to nummerical errors.")
                st.info("The equations above assume that the output can be written as an affine "
                        "function of the input. This is for only for linear systems the case. "
                        "Thus there is a bigger difference if tried on non-linear systems. ")

                st.markdown("Also the first x_dim pca components will explain 100% of the variance"
                            ", i.e. the data in the high dimensional r_dim space spans the same "
                            "dimensions as the input: ")
                cols = st.columns(2)
                with cols[0]:
                    st.write("Explained variance: ", pca_esn._pca.explained_variance_ratio_.T)
                with cols[1]:
                    st.write("Cummulative explained variance: ",
                             np.cumsum(pca_esn._pca.explained_variance_ratio_).T)

            utils.st_line()
            if st.checkbox("Check linear ESN with network: ", key="theory_linear_net"):
                st.markdown("**Theory:**")
                st.markdown("Here we assume a linear activation function with a network. "
                            "Furthermore we assume a output-bias readout and no node bias. "
                            "With these assumptions the output becomes a linear function of all"
                            "previous inputs. The ESN becomes a VAR (vector auto-regression model).")

                st.latex(r"""
                \begin{aligned}
                \boldsymbol{y}_i &= \tilde{W}_\text{out} \tilde{\boldsymbol{r}}_i =  W_\text{out} 
                \boldsymbol{r}_i + \boldsymbol{w} = 
                W_\text{out}\left[ W_\text{in} 
                \boldsymbol{x}_{i-1} + A \boldsymbol{r}_{i-1} + \boldsymbol{b} \right] + \boldsymbol{w} \\ 
                \boldsymbol{y}_i &= W_\text{out} \left[ W_\text{in} 
                \boldsymbol{x}_{i-1} + A W_\text{in} \boldsymbol{x}_{i-2} + A^2 \boldsymbol{r}_{i-2}
                + (A+1) \boldsymbol{b} \right] + \boldsymbol{w} \\
                \boldsymbol{y}_i &= \sum_{j=1} C_j \boldsymbol{x}_{i-j} + \tilde{\boldsymbol{w}}, 
                \qquad C_j \text{ is a } x_\text{dim} \times x_\text{dim} \text{ matrix.} 
                \end{aligned}
                """)

                st.markdown("For the pca esn the output looks like: ")
                st.latex(r"""
                \begin{aligned}
                \boldsymbol{y}_i &= 
                \tilde{W}^\text{pca}_\text{out} \tilde{\boldsymbol{r}}_\text{pca, i} = 
                W_\text{out}^\text{pca} \boldsymbol{r}_\text{pca, i} + \boldsymbol{w}_\text{pca} = 
                W_\text{out}^\text{pca} P (\boldsymbol{r}_i - \boldsymbol{r}_\text{mean}) + 
                \boldsymbol{w}_\text{pca} \\
                
                \boldsymbol{y} &= W_\text{out}^\text{pca} P \boldsymbol{r}_i + \boldsymbol{c} = 
                W_\text{out}^\text{pca} P \left[ W_\text{in} 
                \boldsymbol{x}_{i-1} + A \boldsymbol{r}_{i-1} + \boldsymbol{b} \right] + \boldsymbol{c}\\
                \end{aligned}
                """)

                st.markdown("By comparing both equations we get the following relationships: ")
                st.latex(r"""
                \begin{aligned}
                W_\text{out}^\text{pca} P &= W_\text{out} \\
                
                \boldsymbol{w}_\text{pca} - 
                W_\text{out}^\text{pca} P \boldsymbol{r}_\text{mean} &= \boldsymbol{w}
                \end{aligned}
                """)

                st.markdown("These two relationships will now be tested numerically:")
                st.info(
                    "Please select r_to_r_gen_opt = output_bias, act_fct_opt=linear, node_bias_opt=random_bias,"
                    ", n_rad=0.5")

                # normal esn:
                normal_esn = esn.build(esn_type="ESN_normal",
                                       seed=seed,
                                       x_dim=x_dim,
                                       **build_args)
                r_dim = build_args["r_dim"]
                normal_esn = copy.deepcopy(normal_esn)
                _, _, _, normal_esn = esn.train_return_res(
                    normal_esn,
                    x_train,
                    t_train_sync,
                )
                w_out_normal = normal_esn.get_w_out()
                w_out_res_normal = w_out_normal[:, :r_dim]
                w_out_vec_normal = w_out_normal[:, -1]
                b_normal = normal_esn._node_bias

                # PCA esn:
                pca_esn = esn.build(esn_type="ESN_pca",
                                       seed=seed,
                                       x_dim=x_dim,
                                       **build_args)
                pca_esn = copy.deepcopy(pca_esn)
                _, _, pca_res_dict, pca_esn = esn.train_return_res(
                    pca_esn,
                    x_train,
                    t_train_sync,
                )
                w_out_pca = pca_esn.get_w_out()
                w_out_res_pca = w_out_pca[:, :r_dim]
                w_out_vec_pca = w_out_pca[:, -1]
                r_mean = np.mean(pca_res_dict["r"], axis=0)
                pca_components = pca_esn._pca.components_

                # first test:  w_out matrices
                pca_matrix = w_out_res_pca @ pca_components
                normal_matrix = w_out_res_normal

                st.write("Difference of matrices: ", pca_matrix - normal_matrix)

                # second test:
                pca_vector = w_out_vec_pca - w_out_res_pca @ pca_components @ r_mean
                normal_vector = w_out_vec_normal
                st.write("Differences of vectors: ", pca_vector - normal_vector)


            utils.st_line()
            if st.checkbox("Moment Matrix: ",
                           key="moment matrix"):
                st.markdown("**Theory:**")
                st.markdown("See how ill the moment matrix is conditioned. ")
                r_gen_states = res_train_dict["r_gen"]
                moment_matrix = r_gen_states.T @ r_gen_states
                st.write(moment_matrix)
                # st.write(moment_matrix > 1e-14)

                matrix_rank = np.linalg.matrix_rank(r_gen_states)
                st.write(matrix_rank)

                # st.write("Try to inverse the moment_matrix: ")
                inv_moment_matrix = np.linalg.inv(moment_matrix)
                # Idea: Fast training by just inverting the diagonal?
                st.write("Inverse: ", inv_moment_matrix)

                st.write("Condition of design matrix: ")
                cond = np.linalg.cond(r_gen_states)
                st.write("Condition: ", cond)

                # Do i have to care about multi-colinearity?
                # SEE: https://en.wikipedia.org/wiki/Multicollinearity -> Detection

                # Add noise to data and re-run regression, see how much the coefficients change.
        else:
            st.info('Activate [🔮 Predict] checkbox to see something.')
    #  Container code at the end:
    if build_bool:
        x_dim, r_dim, r_gen_dim, y_dim = esn_obj.get_dimensions()
        with architecture_container:
            esnplot.st_plot_architecture(x_dim=x_dim, r_dim=r_dim, r_gen_dim=r_gen_dim,
                                         y_dim=y_dim)
