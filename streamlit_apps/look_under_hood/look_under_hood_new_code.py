import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import rescomp
import rescomp.esn_new_update_code as esn_new
import rescomp.plotting as plot
import rescomp.statistical_tests as stat_test
plt.style.use('dark_background')


def save_to_yaml(parameter_dict):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
    with open(f'saved_parameters/{dt_string}.yml', 'w') as outfile:
        yaml.dump(parameter_dict, outfile, default_flow_style=False)


@st.experimental_memo
def get_random_int():
    print("Get new seed")
    return np.random.randint(1, 1000000)


@st.experimental_memo
def simulate_data(system_option, dt, all_time_steps, normalize):
    print("simulate data")
    if system_option == "Lorenz":
        starting_point = np.array([0, 1, 0])
        time_series = rescomp.simulations.simulate_trajectory("lorenz", dt, all_time_steps, starting_point)
    elif system_option == "Roessler":
        starting_point = np.array([0, 1, 0])
        time_series = rescomp.simulations.simulate_trajectory("roessler_sprott", dt, all_time_steps,
                                                              starting_point)
    if normalize:
        time_series = rescomp.utilities.normalize_timeseries(time_series)
    return time_series


@st.experimental_singleton
def build(esntype, seed, **kwargs):
    if esntype == "normal":
        esn = esn_new.ESN_normal()
    elif esntype == "dynsys":
        esn = esn_new.ESN_dynsys()
    elif esntype == "difference":
        esn = esn_new.ESN_difference()

    x_dim = 3
    np.random.seed(seed)
    esn.build(x_dim, **kwargs)
    return esn


@st.cache(hash_funcs={rescomp.esn_new_update_code.ESN_normal: hash,
                      rescomp.esn_new_update_code.ESN_dynsys: hash,
                      rescomp.esn_new_update_code.ESN_difference: hash,
                      rescomp.esn_new_update_code.ESN_output_hybrid: hash})
def train(esn, x_train, t_train_sync):
    print("train")
    print("wout: ", esn._w_out)
    esn.train(x_train, sync_steps=t_train_sync, save_res_inp=True, save_r_internal=True, save_r=True,
              save_r_gen=True, save_out=True, save_y_train=True)

    # x_train_true = x_train[1+t_train_sync:]
    act_fct_inp_train = esn.get_act_fct_inp()
    r_internal_train = esn.get_r_internal()
    r_input_train = esn.get_res_inp()
    r_train = esn.get_r()
    x_train_true = esn.get_y_train()
    # r_gen_train = esn.get_r_gen()
    x_train_pred = esn.get_out()
    print("shapes: ", x_train_true.shape, x_train_pred.shape)
    return esn, x_train_true, x_train_pred, r_train, act_fct_inp_train, r_internal_train, r_input_train

@st.cache(hash_funcs={rescomp.esn_new_update_code.ESN_normal: hash,
                      rescomp.esn_new_update_code.ESN_dynsys: hash,
                      rescomp.esn_new_update_code.ESN_difference: hash,
                      rescomp.esn_new_update_code.ESN_output_hybrid: hash})
def predict(esn, x_pred, t_pred_sync):
    print("predict rc")
    y_pred, y_true = esn.predict(x_pred, sync_steps=t_pred_sync, save_res_inp=True, save_r_internal=True, save_r=True)
    r_pred = esn.get_r()
    act_fct_inp_pred = esn.get_act_fct_inp()
    r_internal_pred = esn.get_r_internal()
    r_input_pred = esn.get_res_inp()
    return y_pred, y_true, r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred


# CASHING PLOT FUNCTIONS:
@st.experimental_memo
def plot_original_attractor(time_series):
    print("replotting 1")
    fig = plot.plot_3d_time_series(time_series)
    return fig


@st.experimental_memo
def plot_original_timeseries(time_series, i_dim, time_boundaries):
    print("replotting 2")
    fig = plot.plot_1d_time_series(time_series, i_dim, boundaries=time_boundaries)
    return fig


@st.experimental_memo
def plot_esn_architecture(w_in):
    print("replotting 3")
    fig = plot.plot_architecture(w_in, figsize=(10, 4))
    return fig


@st.experimental_memo
def plot_train_pred_vs_true_trajectories(x_train_true, x_train_pred):
    print("replotting 4")
    figs = []
    for i in range(x_dim):
        title = f"Axis {i}:"
        data = {"Train True": x_train_true[:, i], "Train Fit": x_train_pred[:, i]}
        fig = plot.plot_multiple_1d_time_series(data, title=title)
        figs.append(fig)
    return figs


@st.experimental_memo
def plot_train_pred_vs_true_attractor(x_train_true, x_train_pred):
    print("replotting 5")
    time_series_dict = {"Train True": x_train_true, "Train Fit": x_train_pred}
    fig = plot.plot_3d_time_series_multiple(time_series_dict)
    return fig


@st.experimental_memo
def plot_train_error(x_train_pred, x_train_true):
    print("replotting 6")
    fig = plot.plot_error_single(x_train_pred, x_train_true, title="Train Error")
    return fig


@st.experimental_memo
def plot_reservoir_histograms_train(r_input_train, r_internal_train,
                                    act_fct_inp_train, r_train):
    print("replotting 7")
    states_data_dict = {"r_input": r_input_train, "r_internal_update": r_internal_train,
                        "act_fct_inp": act_fct_inp_train, "r_states": r_train}
    fig = plot.plot_node_value_histogram_multiple(states_data_dict)
    return fig


@st.experimental_memo
def plot_prediction_pred_vs_true_trajectories(y_true, y_pred):
    print("replotting 8")
    figs = []
    for i in range(x_dim):
        title = f"Axis {i}:"
        data = {"True": y_true[:, i], "Predicted": y_pred[:, i]}
        fig = plot.plot_multiple_1d_time_series(data, title=title)
        figs.append(fig)
    return figs


@st.experimental_memo
def plot_prediction_pred_vs_true_attractor(y_true, y_pred):
    print("replotting 9")
    time_series_dict = {"True": y_true, "Predicted": y_pred}
    fig = plot.plot_3d_time_series_multiple(time_series_dict)
    return fig


@st.experimental_memo
def plot_pred_error(y_pred, y_true):
    print("replotting 10")
    fig = plot.plot_error_single(y_pred, y_true, title="Prediction Error")
    return fig


@st.experimental_memo
def plot_reservoir_histograms_pred(r_input_pred, r_internal_pred,
                                    act_fct_inp_pred, r_pred):
    print("replotting 11")
    states_data_dict = {"r_input": r_input_pred, "r_internal_update": r_internal_pred,
                        "act_fct_inp": act_fct_inp_pred, "r_states": r_pred}
    fig = plot.plot_node_value_histogram_multiple(states_data_dict)
    return fig


@st.experimental_memo
def plot_reservoir_histograms_pred(r_input_pred, r_internal_pred, r_pred, y_pred):
    fig = plot.plot_image_and_timeseries(r_input_pred, r_internal_pred, r_pred, y_pred, figsize=(13, 25))
    return fig


esn_types = ["normal", "dynsys", "difference"]
systems_to_predict = ["Lorenz", "Roessler"]
w_in_types = ["ordered_sparse", "random_sparse", "random_dense_uniform", "random_dense_gaussian"]
bias_types = ["no_bias", "random_bias"]
network_types = ["erdos_renyi", "scale_free", "small_world", "random_directed"]
activation_functions = ["tanh", "sigmoid"]
r_to_r_gen_types = ["linear_r", "linear_and_square_r", "output_bias", "bias_and_square_r", "linear_and_square_r_alt"]
dyn_sys_types = ["L96", "KS"]


with st.sidebar:
    # System to predict:
    st.header("System: ")
    # sim_opts = {}
    # sim_opts["system"] = st.sidebar.selectbox(
    #     'System to Predict', systems_to_predict)
    # dt = st.number_input('dt', value=0.01, step=0.01)
    # sim_opts["dt"] = dt
    # with st.expander("Time Steps"):
    #     t_train_disc = int(st.number_input('t_train_disc', value=1000, step=1))
    #     t_train_sync = int(st.number_input('t_train_sync', value=300, step=1))
    #     t_train = int(st.number_input('t_train', value=1000, step=1))
    #     t_pred_disc = int(st.number_input('t_pred_disc', value=1000, step=1))
    #     t_pred_sync = int(st.number_input('t_pred_sync', value=300, step=1))
    #     t_pred = int(st.number_input('t_pred', value=1000, step=1))
    #     all_time_steps = int(t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred)
    #     time_boundaries = (0, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred)
    #     st.write(f"Total Timessteps: {all_time_steps}, Maximal Time: {np.round(dt*all_time_steps, 1)}")
    # sim_opts["normalize"] = st.checkbox('normalize')

    system_option = st.sidebar.selectbox(
        'System to Predict', systems_to_predict)
    dt = st.number_input('dt', value=0.01, step=0.01)
    with st.expander("Time Steps"):
        t_train_disc = int(st.number_input('t_train_disc', value=1000, step=1))
        t_train_sync = int(st.number_input('t_train_sync', value=300, step=1))
        t_train = int(st.number_input('t_train', value=1000, step=1))
        t_pred_disc = int(st.number_input('t_pred_disc', value=1000, step=1))
        t_pred_sync = int(st.number_input('t_pred_sync', value=300, step=1))
        t_pred = int(st.number_input('t_pred', value=1000, step=1))
        all_time_steps = int(t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred)
        time_boundaries = (0, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred)
        st.write(f"Total Timessteps: {all_time_steps}, Maximal Time: {np.round(dt*all_time_steps, 1)}")
    normalize = st.checkbox('normalize')

    sim_opts = {"system": system_option, "dt": dt, "t_train_disc": t_train_disc, "t_train_sync": t_train_sync,
                "t_train": t_train, "t_pred_disc": t_pred_disc, "t_pred_sync": t_pred_sync, "t_pred": t_pred}

    st.markdown("""---""")

    # Build RC architecture:
    st.header("Reservoir Architecture: ")
    esn_type = st.selectbox('esn type', esn_types)

    build_args = {}
    build_args["r_dim"] = st.number_input('Reservoir Dim', value=200, step=1)
    build_args["r_to_r_gen_opt"] = st.selectbox('r_to_r_gen_opt', r_to_r_gen_types)
    build_args["act_fct_opt"] = st.selectbox('act_fct_opt', activation_functions)
    build_args["node_bias_opt"] = st.selectbox('node_bias_opt', bias_types)
    disabled = True if build_args["node_bias_opt"] == "no_bias" else False
    build_args["bias_scale"] = st.number_input('bias_scale', value=0.1, step=0.1, disabled=disabled)  # disable if needed
    build_args["leak_factor"] = st.number_input('leak_factor', value=0.0, step=0.01, min_value=0.0, max_value=1.0)
    build_args["w_in_opt"] = st.selectbox('w_in_opt', w_in_types)
    build_args["w_in_scale"] = st.number_input('w_in_scale', value=1.0, step=0.1)
    log_reg_param = st.number_input('Log regulation parameter', value=-5, step=1)
    build_args["reg_param"] = 10**(log_reg_param)

    # settings depending on esn_type:
    with st.expander("ESN type specific settings:"):
        if esn_type in ["normal", "difference"]:
            # network:
            build_args["n_rad"] = st.number_input('n_rad', value=0.1, step=0.1)
            build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=5.0, step=0.1)
            build_args["n_type_opt"] = st.selectbox('n_type_opt', network_types)
            if esn_type == "difference":
                build_args["dt_difference"] = st.number_input('dt_difference', value=0.1, step=0.01)
        elif esn_type == "dynsys":
            build_args["dyn_sys_opt"] = st.selectbox('dyn_sys_opt', dyn_sys_types)
            build_args["dyn_sys_dt"] = st.number_input('dyn_sys_dt', value=0.1, step=0.01)
            build_args["scale_factor"] = st.number_input('scale_factor', value=1.0, step=0.1)
            custom_dyn_sys_parameter = st.number_input('custom dyn_sys parameter', value=5.0, step=0.1)  # make dependent on case
            build_args["dyn_sys_other_params"] = (custom_dyn_sys_parameter, )
    st.markdown("""---""")

    if st.button("new seed"):
        get_random_int.clear()
    seed = get_random_int()  # lets see what to do with that
    st.write(f"Current seed: {seed}")

    st.markdown("""---""")

    if st.button("Download Parameters"):
        parameter_dict = {"sim_opts": sim_opts, "esn_type": esn_type, "build_parameters": build_args, "seed": seed}  # add seed?
        save_to_yaml(parameter_dict)

# Timeseries
with st.expander("Simulate Timeseries"):
    st.header("Time series:")
    simulate_timeseries_bool = st.checkbox("Simulate Timeseries")
    if simulate_timeseries_bool:
        time_series = simulate_data(system_option, dt, all_time_steps, normalize)
        x_dim = time_series.shape[1]
        plot_attractor_time_series = st.checkbox("Plot Attractor")
        if plot_attractor_time_series:
            fig = plot_original_attractor(time_series)
            st.plotly_chart(fig, use_container_width=True)
            # fig = plot.plot_3d_time_series(time_series)
            # st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            plot_trajectory_time_series = st.checkbox("Plot Trajectory")
        with right:
            i_dim = st.number_input("i_dim", min_value=0, max_value=x_dim-1)

        if plot_trajectory_time_series:
            fig = plot_original_timeseries(time_series, i_dim, time_boundaries)
            # fig = plot.plot_1d_time_series(time_series, i_dim, boundaries=time_boundaries)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")

# RC Architecture:
disabled = False if simulate_timeseries_bool else True
with st.expander("RC architecture"):
    st.header("RC architecture")
    build_rc_bool = st.checkbox("Build RC architecture", disabled=disabled)
    if build_rc_bool:

        # if st.button("rebuild with new seed"):
        #     get_random_int.clear()
        #
        # seed = get_random_int()  # lets see what to do with that

        esn = build(esn_type, seed, **build_args)

        w_in_properties_bool = st.checkbox("Show W_in properties:")
        if w_in_properties_bool:
            # fig = plot.plot_architecture(esn, figsize=(10, 4))
            w_in = esn._w_in
            fig = plot_esn_architecture(w_in)
            st.pyplot(fig)

        # if custom_update_setting_str == "Normal":
        #     w_properties_bool = st.checkbox("Show W properties:")
        #     if w_properties_bool:
        #         st.write("TODO: Show Network, Histograms of values, some measures?")
        st.markdown("""---""")

# Train RC:
disabled = False if build_rc_bool else True
with st.expander("Train RC"):
    st.header("Train RC: ")
    train_bool = st.checkbox("Train RC", disabled=disabled)
    if train_bool:
        x_train, x_pred_list = stat_test.data_simulation(time_series, t_train_disc, t_train_sync, t_train,
                                                         t_pred_disc, t_pred_sync, t_pred, nr_of_time_intervals=1,
                                                         sim_data_return=False, v=0)
        esn, x_train_true, x_train_pred, r_train, act_fct_inp_train, r_internal_train, r_input_train = train(esn,
                                                                                                             x_train,
                                                                                                 t_train_sync)

        show_x_train_pred = st.checkbox("Show fitted and real trajectory:")
        if show_x_train_pred:
            figs = plot_train_pred_vs_true_trajectories(x_train_true, x_train_pred)
            for i in range(x_dim):
                st.plotly_chart(figs[i])

        show_x_train_attr = st.checkbox("Show fitted and real attractor:")
        if show_x_train_attr:
            # time_series_dict = {"Train True": x_train_true, "Train Fit": x_train_pred}
            # fig = plot.plot_3d_time_series_multiple(time_series_dict)
            fig = plot_train_pred_vs_true_attractor(x_train_true, x_train_pred)
            st.plotly_chart(fig)

        show_train_error = st.checkbox("Show train error:")
        if show_train_error:
            fig = plot_train_error(x_train_pred, x_train_true)
            st.plotly_chart(fig)

        # show_reservoir = st.checkbox("Show Reservoir states: ")
        # if show_reservoir:
        #     fig = plot.show_reservoir_states(r_train)
        #     st.plotly_chart(fig)

        show_reservoir_histograms_train = st.checkbox("Show Reservoir node value histograms - TRAIN: ")
        if show_reservoir_histograms_train:
            # states_data_dict = {"r_input": r_input_train, "r_internal_update": r_internal_train,
            #                     "act_fct_inp": act_fct_inp_train, "r_states": r_train}
            # fig = plot.plot_node_value_histogram_multiple(states_data_dict)
            fig = plot_reservoir_histograms_train(r_input_train, r_internal_train,
                                    act_fct_inp_train, r_train)
            st.pyplot(fig)

    st.markdown("""---""")

# Predict with RC:
disabled = False if train_bool else True
with st.expander("Prediction"):
    st.header("Predict with RC: ")
    predict_bool = st.checkbox("Predict with RC", disabled=disabled)
    if predict_bool:
        x_pred = x_pred_list[0]
        y_pred, y_true, r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred = predict(esn, x_pred, t_pred_sync)
        show_x_pred = st.checkbox("Show predicted and real trajectory:")
        if show_x_pred:
            figs = plot_prediction_pred_vs_true_trajectories(y_true, y_pred)
            for i in range(x_dim):
                st.plotly_chart(figs[i])
                # title = f"Axis {i}:"
                # data = {"True": y_true[:, i], "Predicted": y_pred[:, i]}
                # fig = plot.plot_multiple_1d_time_series(data, title=title)
                # st.plotly_chart(fig)

        show_x_pred_attr = st.checkbox("Show predicted and real attractor:")
        if show_x_pred_attr:
            # time_series_dict = {"True": y_true, "Predicted": y_pred}
            # fig = plot.plot_3d_time_series_multiple(time_series_dict)
            fig = plot_prediction_pred_vs_true_attractor(y_true, y_pred)
            st.plotly_chart(fig)

        show_pred_error = st.checkbox("Show predicted error:")
        if show_pred_error:
            # fig = plot.plot_error_single(y_pred, y_true, title="Prediction Error")
            fig = plot_pred_error(y_pred, y_true)
            st.plotly_chart(fig)

        # show_r_pred = st.checkbox("Show reservoir states for Prediction:")
        # if show_r_pred:
        #     fig = plot.show_reservoir_states(r_pred)
        #     st.plotly_chart(fig)

        show_reservoir_histograms_pred = st.checkbox("Show Reservoir node value histograms - PRED: ")
        if show_reservoir_histograms_pred:
            # states_data_dict = {"r_input": r_input_pred, "r_internal_update": r_internal_pred,
            #                     "act_fct_inp": act_fct_inp_pred, "r_states": r_pred}
            # fig = plot.plot_node_value_histogram_multiple(states_data_dict)
            fig = plot_reservoir_histograms_pred(r_input_pred, r_internal_pred, act_fct_inp_pred, r_pred)
            st.pyplot(fig)

        show_trajectory_and_resstates = st.checkbox("Show Reservoir states and trajectories - PRED: ")
        if show_trajectory_and_resstates:
            # fig = plot.plot_image_and_timeseries(r_input_pred, r_internal_pred, r_pred, y_pred, figsize=(13, 25))
            fig = plot_reservoir_histograms_pred(r_input_pred, r_internal_pred, r_pred, y_pred)
            st.pyplot(fig)
    st.markdown("""---""")
