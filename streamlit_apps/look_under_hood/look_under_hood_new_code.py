import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rescomp
import rescomp.esn_new_update_code as esn_new
import rescomp.plotting as plot
import rescomp.statistical_tests as stat_test
plt.style.use('dark_background')


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


# Functions:

@st.experimental_singleton
def build(esntype, **kwargs):
    if esntype == "normal":
        esn = esn_new.ESN_normal()
    elif esntype == "dynsys":
        esn = esn_new.ESN_dynsys()
    elif esntype == "difference":
        esn = esn_new.ESN_difference()

    x_dim = 3
    esn.build(x_dim, **kwargs)


@st.cache(hash_funcs={rescomp.esn_new_update_code.ESN_normal: hash,
                      rescomp.esn_new_update_code.ESN_dynsys: hash,
                      rescomp.esn_new_update_code.ESN_difference: hash,
                      rescomp.esn_new_update_code.ESN_output_hybrid: hash})
def train(esn, x_train, t_train_sync):
    print("train")
    esn.train(x_train, sync_steps=t_train_sync, save_res_inp=True, save_r_internal=True, save_r=True,
              save_r_gen=True, save_out=True)

    x_train_true = x_train[1+t_train_sync:]
    act_fct_inp_train = esn.get_act_fct_inp()
    r_internal = esn.get_r_internal()
    r_train = esn.get_r()
    r_gen_train = esn.get_r_gen()
    x_train_pred = esn.get_out()

    return esn, x_train_true, x_train_pred, r_train

@st.cache(hash_funcs={rescomp.esn_new_update_code.ESN_normal: hash,
                      rescomp.esn_new_update_code.ESN_dynsys: hash,
                      rescomp.esn_new_update_code.ESN_difference: hash,
                      rescomp.esn_new_update_code.ESN_output_hybrid: hash})
def predict_rc(esn, x_pred, t_pred_sync):
    print("predict rc")
    y_pred, y_true = esn.predict(x_pred=x_pred, sync_steps=t_pred_sync, save_r=True)
    r_pred = esn.get_r()
    return y_pred, y_true, r_pred

#
# @st.cache(hash_funcs={rescomp.ESNWrapper: hash})
# def predict_rc(esn, x_pred, t_pred_sync):
#     print("predict rc")
#     esn.reset_res_state()
#     y_pred, y_true = esn.predict(x_pred=x_pred, sync_steps=t_pred_sync, save_r=True)
#     r_pred = esn._r_pred_gen
#     return y_pred, y_true, r_pred


w_in_types = ["ordered_sparse", "random_sparse", "random_dense", "rect", "gauss",
              "alternating", "sinus"]
systems_to_predict = ["Lorenz", "Roessler"]
activation_functions = ["tanh", "sigmoid"]
w_out_fit_flags = ["simple", "linear_and_square_r", "bias", "bias_and_square_r",
                   "linear_and_square_r_alt"]

with st.sidebar:
    # System to predict:
    st.header("System: ")
    system_option = st.sidebar.selectbox(
        'System to Predict', systems_to_predict)
    dt = st.number_input('dt', value=0.01, step=0.01)
    with st.expander("Time Steps"):
        t_train_disc = int(st.number_input('t_train_disc', value=1000, step=1))
        t_train_sync = int(st.number_input('t_train_sync', value=200, step=1))
        t_train = int(st.number_input('t_train', value=1000, step=1))
        t_pred_disc = int(st.number_input('t_pred_disc', value=1000, step=1))
        t_pred_sync = int(st.number_input('t_pred_sync', value=100, step=1))
        t_pred = int(st.number_input('t_pred', value=1000, step=1))
        all_time_steps = int(t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred)
        time_boundaries = (0, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred)
        st.write(f"Total Timessteps: {all_time_steps}, Maximal Time: {np.round(dt*all_time_steps, 1)}")
    normalize = st.checkbox('normalize')
    st.markdown("""---""")

    # Build RC architecture:
    st.header("Reservoir Architecture: ")
    r_dim = st.number_input('Reservoir Dim', value=200, step=1)
    w_in_scale = st.number_input('w_in_scale', value=1.0, step=0.1)
    w_in_type = st.selectbox('w_in_type', w_in_types)
    activation_function = st.selectbox('activation_function', activation_functions)
    w_out_fit_flag = st.selectbox('w_out_fit_flag', w_out_fit_flags)
    leak_fct = st.number_input('leak factor', value=0.0, step=0.01, min_value=0.0, max_value=1.0)

    log_reg_param = st.number_input('Log regulation parameter', value=-5, step=1)
    reg_param = 10**(log_reg_param)

    # custom update:
    custom_update_settings = ["Normal", "L96", "KS"]
    custom_update_setting_str = st.selectbox('custom_update_settings', custom_update_settings)
    if custom_update_setting_str == "Normal":
        with st.expander("Adjecency Matrix settings: "):
            n_rad = st.number_input('n_rad', value=0.1, step=0.1)
            n_avg_deg = st.number_input('n_avg_deg', value=6.0, step=0.1)
            custom_args = (custom_update_setting_str, n_rad, n_avg_deg)
    elif custom_update_setting_str == "KS":
        with st.expander("KS System Settings: "):
            dt_custom = st.number_input('dt_KS', value=0.5, step=0.01)
            L_KS = st.number_input('KS systemsize', value=40, step=2)
            custom_args = (custom_update_setting_str, dt_custom, L_KS)
    elif custom_update_setting_str == "L96":
        with st.expander("L96 System Settings: "):
            dt_custom = st.number_input('dt_L96', value=0.5, step=0.01)
            L96_Force = st.number_input('L95 Force', value=5, step=1)
            custom_args = (custom_update_setting_str, dt_custom, L96_Force)

    st.markdown("""---""")

# Timeseries
with st.container():
    st.header("Time series:")
    simulate_timeseries_bool = st.checkbox("Simulate Timeseries")
    if simulate_timeseries_bool:
        time_series = simulate_data(system_option, dt, all_time_steps, normalize)
        x_dim = time_series.shape[1]
        plot_attractor_time_series = st.checkbox("Plot Attractor")
        if plot_attractor_time_series:
            fig = plot.plot_3d_time_series(time_series)
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            plot_trajectory_time_series = st.checkbox("Plot Trajectory")
        with right:
            i_dim = st.number_input("i_dim", min_value=0, max_value=x_dim-1)

        if plot_trajectory_time_series:
            fig = plot.plot_1d_time_series(time_series, i_dim, boundaries=time_boundaries)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""---""")

# RC Architecture:
disabled = False if simulate_timeseries_bool else True
with st.container():
    st.header("RC architecture")
    build_rc_bool = st.checkbox("Build RC architecture", disabled=disabled)
    if build_rc_bool:

        if st.button("rebuild with new seed"):
            get_random_int.clear()

        seed = get_random_int()
        build_args = (r_dim, x_dim, w_in_scale, w_in_type, activation_function,
                      w_out_fit_flag, leak_fct, custom_args, seed)
        esn = build_rc(*build_args)

        w_in_properties_bool = st.checkbox("Show W_in properties:")
        if w_in_properties_bool:
            fig = plot.plot_architecture(esn, figsize=(10, 4))
            st.pyplot(fig)

        if custom_update_setting_str == "Normal":
            w_properties_bool = st.checkbox("Show W properties:")
            if w_properties_bool:
                st.write("TODO: Show Network, Histograms of values, some measures?")
        st.markdown("""---""")

# Train RC:
disabled = False if build_rc_bool else True
with st.container():
    st.header("Train RC: ")
    train_bool = st.checkbox("Train RC", disabled=disabled)
    if train_bool:
        x_train, x_pred_list = stat_test.data_simulation(time_series, t_train_disc, t_train_sync, t_train,
                                                         t_pred_disc, t_pred_sync, t_pred, nr_of_time_intervals=1,
                                                         sim_data_return=False, v=0)
        esn, x_train_true, x_train_pred, r_train = train_rc(esn, x_train, t_train_sync, reg_param)

        show_x_train_pred = st.checkbox("Show fitted and real trajectory:")
        if show_x_train_pred:
            for i in range(x_dim):
                title = f"Axis {i}:"
                data = {"train fit": x_train_pred[:, i], "train true": x_train_true[:, i]}
                fig = plot.plot_multiple_1d_time_series(data, title=title)
                st.plotly_chart(fig)
        show_reservoir = st.checkbox("Show Reservoir states: ")
        if show_reservoir:
            fig = plot.show_reservoir_states(r_train)
            st.plotly_chart(fig)

    st.markdown("""---""")

# Predict with RC:
disabled = False if train_bool else True
with st.container():
    st.header("Predict with RC: ")
    predict_bool = st.checkbox("Predict with RC", disabled=disabled)
    if predict_bool:
        x_pred = x_pred_list[0]
        y_pred, y_true, r_pred = predict_rc(esn, x_pred, t_pred_sync)
        show_x_pred = st.checkbox("Show predicted and real trajectory:")
        if show_x_pred:
            for i in range(x_dim):
                title = f"Axis {i}:"
                data = {"True": y_true[:, i], "Predicted": y_pred[:, i]}
                fig = plot.plot_multiple_1d_time_series(data, title=title)
                st.plotly_chart(fig)
        show_r_pred = st.checkbox("Show reservoir states for Prediction:")
        if show_r_pred:
            fig = plot.show_reservoir_states(r_pred)
            st.plotly_chart(fig)
    st.markdown("""---""")
