import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
from sklearn import decomposition
import pandas as pd
import plotly.express as px

import rescomp
import rescomp.esn_new_update_code as esn_new
import rescomp.plotting as plot
import rescomp.statistical_tests as stat_test
plt.style.use('dark_background')

lyapunov_exponents = rescomp.simulations.standard_lyapunov_exponents
lyapunov_exponents_extra = {"lorenz_plus_roessler": None,
                      "logistic": None, "lorenz96": None, "periodic": None}
lyapunov_exponents.update(lyapunov_exponents_extra)

starting_points = rescomp.simulations.standard_starting_points
starting_points_extra = {"lorenz_plus_roessler": None, "logistic": None,
                         "lorenz96": None, "periodic": None}
starting_points.update(starting_points_extra)

# lyapunov_exponents = {"lorenz": 0.9056, "roessler": 0.0714, "chua": 0.3271, "chen": 2.027, "complex_butterfly": 0.1690,
#                       "rucklidge": 0.0643, "thomas": 0.0349, "windmi": 0.0755, "lorenz_plus_roessler": None,
#                       "logistic": None, "lorenz96": None, "simplest_quadratic": 0.0551, "simplest_cubic": 0.0837,
#                       "simplest_piecewise": 0.0362,
#                       }
#
# starting_points = {"lorenz": np.array([0, -0.01, 9]), "roessler": np.array([-9, 0, 0]),
#                    "chua": np.array([0, 0, 0.6]), "chen": np.array([-10, 0, 37]),
#                    "complex_butterfly": np.array([0.2, 0.0, 0.0]),
#                    "rucklidge": np.array([1.0, 0.0, 4.5]), "thomas": np.array([0.1, 0.0, 0.0]),
#                    "windmi": np.array([0.0, 0.8, 0.0]), "lorenz_plus_roessler": None, "logistic": None,
#                    "lorenz96": None, "simplest_quadratic": np.array([-0.9, 0, 0.5]),
#                    "simplest_cubic": np.array([0.0, 0.96, 0.0]),
#                    "simplest_piecewise": np.array([0.0, -0.7, 0.0]),
#                    }

esn_types = ["normal", "dynsys", "difference", "no_res", "pca", "dynsys_pca", "normal_centered", "pca_noise", "input_to_rgen", "pca_drop"]
systems_to_predict = ["lorenz", "roessler", "chua", "chen", "complex_butterfly", "rucklidge", "thomas", "windmi",
                      "lorenz_plus_roessler", "logistic", "lorenz96", "simplest_quadratic", "simplest_cubic", "simplest_piecewise",
                      "periodic"]
w_in_types = ["random_sparse", "ordered_sparse", "random_dense_uniform", "random_dense_gaussian"]
bias_types = ["no_bias", "random_bias", "constant_bias"]
network_types = ["erdos_renyi", "scale_free", "small_world", "random_directed", "random_dense_gaussian"]
activation_functions = ["tanh", "sigmoid", "relu", "linear"]
r_to_r_gen_types = ["linear_r", "linear_and_square_r", "output_bias", "bias_and_square_r", "linear_and_square_r_alt",
                    "exponential_r", "bias_and_exponential_r"]
dyn_sys_types = ["L96", "KS"]


esn_hash_funcs = {rescomp.esn_new_update_code.ESN_normal: hash,
                  rescomp.esn_new_update_code.ESN_dynsys: hash,
                  rescomp.esn_new_update_code.ESN_difference: hash,
                  rescomp.esn_new_update_code.ESN_output_hybrid: hash,
                  rescomp.esn_new_update_code.ESN_no_res: hash,
                  rescomp.esn_new_update_code.ESN_pca_adv: hash,
                  rescomp.esn_new_update_code.ESN_pca: hash,
                  rescomp.esn_new_update_code.ESN_dynsys_pca: hash,
                  rescomp.esn_new_update_code.ESN_normal_centered: hash,
                  rescomp.esn_new_update_code.ESN_pca_noise: hash,
                  }


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
    starting_point = starting_points[system_option]
    if system_option == "lorenz":
        time_series = rescomp.simulations.simulate_trajectory("lorenz", dt, all_time_steps, starting_point)
    elif system_option == "roessler":
        time_series = rescomp.simulations.simulate_trajectory("roessler_sprott", dt, all_time_steps,
                                                              starting_point)
    elif system_option == "chua":
        time_series = rescomp.simulations.simulate_trajectory("chua", dt, all_time_steps, starting_point)
    elif system_option == "chen":
        time_series = rescomp.simulations.simulate_trajectory("chen", dt, all_time_steps, starting_point)

    elif system_option == "lorenz_plus_roessler":
        starting_point = np.array([0, 1, 0])
        time_series_1 = rescomp.simulations.simulate_trajectory("lorenz", dt, all_time_steps, starting_point)
        time_series_2 = rescomp.simulations.simulate_trajectory("roessler_sprott", dt, all_time_steps, starting_point)
        time_series = time_series_1 + time_series_2

    elif system_option == "logistic":
        starting_point = np.array([0.3])
        r = 4
        time_series = rescomp.simulations.simulate_trajectory(sys_flag="logistic", time_steps=all_time_steps,
                                                              starting_point=starting_point, r=r)
        time_series = time_series - 0.5
        # time_series = time_series[:, np.newaxis]

    elif system_option == "lorenz96":
        starting_point = np.random.randn(30)
        time_series = rescomp.simulations.simulate_trajectory("lorenz_96", dt, all_time_steps,
                                                              starting_point=starting_point)
    elif system_option == "complex_butterfly":
        time_series = rescomp.simulations.simulate_trajectory("complex_butterfly", dt, all_time_steps, starting_point)

    elif system_option == "rucklidge":
        time_series = rescomp.simulations.simulate_trajectory("rucklidge", dt, all_time_steps, starting_point)

    elif system_option == "thomas":
        time_series = rescomp.simulations.simulate_trajectory("thomas", dt, all_time_steps, starting_point)

    elif system_option == "windmi":
        time_series = rescomp.simulations.simulate_trajectory("windmi", dt, all_time_steps, starting_point)

    elif system_option == "simplest_quadratic":
        time_series = rescomp.simulations.simulate_trajectory("simplest_quadratic", dt, all_time_steps, starting_point)
    elif system_option == "simplest_cubic":
        time_series = rescomp.simulations.simulate_trajectory("simplest_cubic", dt, all_time_steps, starting_point)
    elif system_option == "simplest_piecewise":
        time_series = rescomp.simulations.simulate_trajectory("simplest_piecewise", dt, all_time_steps, starting_point)

    elif system_option == "periodic":
        t = np.arange(0, all_time_steps) * dt
        w_1, w_2, w_3 = 0.1, 0.2, 0.3
        a_1, a_2, a_3 = 0.1, 0.5, 0.8
        b_1, b_2, b_3 = 0.1, 0.5, 0.6
        time_series = np.zeros((all_time_steps, 3))
        time_series[:, 0] = a_1 * np.sin(w_1*t + b_1)
        time_series[:, 1] = a_2 * np.sin(w_2*t + b_2)
        time_series[:, 2] = a_3 * np.sin(w_3*t + b_3)

    if normalize:
        time_series = rescomp.utilities.normalize_timeseries(time_series)
    return time_series


@st.experimental_singleton
def build(esntype, seed, x_dim=3, **kwargs):
    if esntype == "normal":
        esn = esn_new.ESN_normal()
    elif esntype == "dynsys":
        esn = esn_new.ESN_dynsys()
    elif esntype == "difference":
        esn = esn_new.ESN_difference()
    elif esntype == "no_res":
        esn = esn_new.ESN_no_res()
    elif esntype == "pca_drop":
        esn = esn_new.ESN_pca_adv()
    elif esntype == "input_to_rgen":
        esn = esn_new.ESN_output_hybrid()  # but dont give a model -> i.e. its just the identiy: f:x -> x
    elif esntype == "pca":
        esn = esn_new.ESN_pca()
    elif esntype == "dynsys_pca":
        esn = esn_new.ESN_dynsys_pca()
    elif esntype == "normal_centered":
        esn = esn_new.ESN_normal_centered()
    elif esntype == "pca_noise":
        esn = esn_new.ESN_pca_noise()

    np.random.seed(seed)
    esn.build(x_dim, **kwargs)
    return esn


@st.cache(hash_funcs=esn_hash_funcs)
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
    r_gen_train = esn.get_r_gen()
    x_train_pred = esn.get_out()
    print("shapes: ", x_train_true.shape, x_train_pred.shape)
    return esn, x_train_true, x_train_pred, r_train, act_fct_inp_train, r_internal_train, r_input_train, r_gen_train


@st.cache(hash_funcs=esn_hash_funcs)
def predict(esn, x_pred, t_pred_sync):
    print("predict rc")
    y_pred, y_true = esn.predict(x_pred, sync_steps=t_pred_sync, save_res_inp=True, save_r_internal=True, save_r=True,
                                 save_r_gen=True)
    r_pred = esn.get_r()
    act_fct_inp_pred = esn.get_act_fct_inp()
    r_internal_pred = esn.get_r_internal()
    r_input_pred = esn.get_res_inp()
    r_gen_pred = esn.get_r_gen()
    return y_pred, y_true, r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred, r_gen_pred


@st.cache(hash_funcs=esn_hash_funcs)
def predict_2(esn, x_pred, t_pred_sync):

    esn.reset_r()

    if t_pred_sync > 0:
        sync = x_pred[:t_pred_sync]
        true_data = x_pred[t_pred_sync:]
        esn.drive(sync, save_r=True)
    else:
        true_data = x_pred

    r_drive = esn.get_r()

    steps = true_data.shape[0]
    y_pred, y_true = esn.loop(steps, save_res_inp=True, save_r_internal=True, save_r=True, save_r_gen=True), \
                     true_data

    r_pred = esn.get_r()
    r_gen_pred = esn.get_r_gen()
    act_fct_inp_pred = esn.get_act_fct_inp()
    r_internal_pred = esn.get_r_internal()
    r_input_pred = esn.get_res_inp()

    return y_pred, y_true, r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred, r_gen_pred, r_drive


@st.cache(hash_funcs=esn_hash_funcs)
def loop_with_perturbation(esn, perturbation, r_pred_previous, time_steps_loop=1000):
    print("run loop")
    print(perturbation)
    # set esn state to first prediction esn state:
    # r_dim = r_pred_previous.size
    # perturbation = np.random.randn(r_dim)
    # pert_len = np.linalg.norm(perturbation)
    # if pert_len != 0:
    #     perturbation = perturbation/pert_len
    # perturbation = perturbation * perturbation_scale

    esn.set_r(r_pred_previous + perturbation)

    esn.loop(time_steps_loop, save_r=True)
    r_loop_perturbed = esn.get_r()
    return r_loop_perturbed

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
def plot_w_out(w_out):
    fig = plot.plot_w_out(w_out, figsize=(10, 4))
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
                                    act_fct_inp_train, r_train, _act_fct):
    print("replotting 7")
    states_data_dict = {"r_input": r_input_train, "r_internal_update": r_internal_train,
                        "act_fct_inp": act_fct_inp_train, "r_states": r_train}
    fig = plot.plot_node_value_histogram_multiple(states_data_dict, act_fct=_act_fct)
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
                                    act_fct_inp_pred, r_pred, _act_fct):
    print("replotting 11")
    states_data_dict = {"r_input": r_input_pred, "r_internal_update": r_internal_pred,
                        "act_fct_inp": act_fct_inp_pred, "r_states": r_pred}
    fig = plot.plot_node_value_histogram_multiple(states_data_dict, act_fct=_act_fct)
    return fig


@st.experimental_memo
def plot_trajectory_and_resstates(r_input_pred, r_internal_pred, r_pred, y_pred):
    fig = plot.plot_image_and_timeseries(r_input_pred, r_internal_pred, r_pred, y_pred, figsize=(13, 25))
    return fig


@st.experimental_memo
def plot_model_likeness():
    pass


@st.experimental_memo
def plot_correlation_dimension(y_pred, y_true, nr_steps, r_min, r_max):
    fig = plot.plot_correlation_dimension(y_pred, y_true, nr_steps=nr_steps, figsize=(13, 5), r_min=r_min, r_max=r_max)
    return fig


@st.experimental_memo
def plot_poincare_type_map(y_pred, y_true, mode="maxima", figsize=(13, 5)):
    fig = plot.plot_poincare_type_map(y_pred, y_true, dim=None, mode=mode, figsize=figsize, alpha=0.2, s=10)
    return fig


@st.experimental_memo
def plot_poincare_type_map_plotly(y_pred, y_true, mode="maxima", value_or_time="value", figsize=(20, 30)):
    fig = plot.plot_poincare_type_map_plotly(y_pred, y_true, dim=None, mode=mode, value_or_time=value_or_time,
                                             figsize=figsize, alpha=0.5, s=3)
    return fig


@st.experimental_memo
def perform_pca(r_train, n_components=3):
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(r_train)
    return pca


@st.experimental_memo
def plot_pca(r_pca, mode="line"):
    data = {"r_pca": r_pca}
    fig = plot.plot_3d_time_series_multiple(data, mode=mode, size=1)
    return fig


@st.experimental_memo
def plot_pca_with_drive(r_pca_drive, r_pca):
    data = {"r_pca_pred": r_pca, "r_pca_drive": r_pca_drive}
    fig = plot.plot_3d_time_series_multiple(data)
    return fig

@st.experimental_memo
def plot_pca_with_perturbed_pca(r_loop_perturbed_pca, r_pred_pca):
    data = {"r_loop_perturbed": r_loop_perturbed_pca, "r_pred_pca": r_pred_pca}
    fig = plot.plot_3d_time_series_multiple(data)
    return fig


@st.experimental_memo
def show_lypunov_from_data(y_pred, y_true, dt, freq_cut=True, steps=100):
    data = {"True": y_true, "Predicted": y_pred}
    fig = plot.plot_lyapunov_spectrum_multiple(data, dt=dt, freq_cut=freq_cut, pnts_to_try=50, steps=steps,
                                               figsize=(13, 5))
    return fig

@st.cache(hash_funcs=esn_hash_funcs)
def plot_pde_difference_vs_pert(esn, r_init, time_steps=2000, n_ens=30, pert_max=2,
                                              pert_min=0.01, pert_steps=20):
    df = rescomp.measures.perturbation_of_res_dynamics(esn, r_init=r_init, time_steps=time_steps, n_ens=n_ens,
                                                       pert_max=pert_max, pert_min=pert_min, pert_steps=pert_steps)
    fig1 = px.line(df, x="pert_scale", y="diff_median", error_y_minus="diff_lower_quartile", error_y="diff_upper_quartile")
    fig2 = px.line(df, x="pert_scale", y="diff_mean")
    return df, fig1, fig2


@st.experimental_memo
def plot_closest_distance_to_attractor(y_pred, y_true):
    closest_dist = rescomp.measures.distances_to_closest_point(y_to_test=y_pred, y_true=y_true)
    fig = plot.plot_multiple_1d_time_series({"closest_distance": closest_dist})
    return fig


@st.experimental_memo
def plot_w_out_and_r_gen_std(w_out, r_gen_train, r_gen_pred):

    fig1 = plot.plot_wout_magnitudes(w_out, figsize=(650, 500))
    fig2 = plot.plot_state_std({"r_gen_train": r_gen_train, "r_gen_pred": r_gen_pred}, figsize=(650, 500),
                          title="std of r_gen during train and pred")
    return fig1, fig2


@st.experimental_memo
def plot_w_out_and_r_gen_std_quantites(r_gen_data, w_out, log_y=False):
    figs = plot.plot_w_out_and_r_gen_std_quantites(r_gen_data, w_out, log_y=log_y)
    return figs

@st.experimental_memo
def plot_valid_times_vs_pred_error(y_pred, y_true, in_lyapunov_times=None):
    return plot.plot_valid_times_vs_pred_error(y_pred, y_true, in_lyapunov_times=in_lyapunov_times)


@st.experimental_memo
def show_reservoir_states(r):
    fig = plot.show_reservoir_states(r)
    return fig


with st.sidebar:
    # System to predict:
    st.header("System: ")

    system_option = st.sidebar.selectbox(
        'System to Predict', systems_to_predict)
    dt = st.number_input('dt', value=0.05, step=0.01, format="%f")
    with st.expander("Time Steps"):
        t_train_disc = int(st.number_input('t_train_disc', value=1000, step=1))
        t_train_sync = int(st.number_input('t_train_sync', value=300, step=1))
        t_train = int(st.number_input('t_train', value=5000, step=1))
        t_pred_disc = int(st.number_input('t_pred_disc', value=1000, step=1))
        t_pred_sync = int(st.number_input('t_pred_sync', value=300, step=1))
        t_pred = int(st.number_input('t_pred', value=5000, step=1))
        all_time_steps = int(t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred)
        time_boundaries = (0, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred)
        st.write(f"Total Timessteps: {all_time_steps}, Maximal Time: {np.round(dt*all_time_steps, 1)}")
        try:
            st.write(f"Lyapunov times: {np.round(dt * all_time_steps * lyapunov_exponents[system_option], 1)}")
        except:
            pass
    normalize = st.checkbox('normalize', value=True)
    train_noise = st.checkbox("train noise")
    noise_scale = st.number_input("noise scale", value=0.1, step=0.1, disabled=not(train_noise))

    sim_opts = {"system": system_option, "dt": dt, "t_train_disc": t_train_disc, "t_train_sync": t_train_sync,
                "t_train": t_train, "t_pred_disc": t_pred_disc, "t_pred_sync": t_pred_sync, "t_pred": t_pred,
                "train_noise": train_noise, "noise_scale": noise_scale}

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
    log_reg_param = st.number_input('Log regulation parameter', value=-7, step=1)
    build_args["reg_param"] = 10**(log_reg_param)

    # settings depending on esn_type:
    with st.expander("ESN type specific settings:"):
        if esn_type in ["normal", "difference", "input_to_rgen", "pca", "normal_centered", "pca_noise"]:
            # network:
            build_args["n_rad"] = st.number_input('n_rad', value=0.1, step=0.1, format="%f")
            build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=5.0, step=0.1)
            build_args["n_type_opt"] = st.selectbox('n_type_opt', network_types)
            if esn_type == "difference":
                build_args["dt_difference"] = st.number_input('dt_difference', value=0.1, step=0.01)
            elif esn_type == "pca" or esn_type == "pca_noise":
                build_args["pca_components"] = int(st.number_input('pca_components', value=build_args["r_dim"], step=1, min_value=1,
                                                                   max_value=int(build_args["r_dim"]), key="pca2"))
                build_args["pca_comps_to_skip"] = int(st.number_input('pca_comps_to_skip', value=0, step=1, min_value=0,
                                                                   max_value=int(build_args["r_dim"])-1))
                left, right = st.columns(2)
                with left:
                    build_args["norm_with_expl_var"] = st.checkbox("norm with explained var", value=False)
                with right:
                    build_args["centering_pre_trans"] = st.checkbox("center data before transformation", value=True)
                if esn_type == "pca_noise":
                    build_args["train_noise_scale"] = st.number_input('train noise scale', value=0.1, step=0.1,
                                                                      min_value=0.0, format="%f")

        elif esn_type in ["dynsys", "dynsys_pca"]:
            build_args["dyn_sys_opt"] = st.selectbox('dyn_sys_opt', dyn_sys_types)
            build_args["dyn_sys_dt"] = st.number_input('dyn_sys_dt', value=0.1, step=0.01)
            build_args["scale_factor"] = st.number_input('scale_factor', value=1.0, step=0.1)
            if build_args["dyn_sys_opt"] == "L96":
                build_args["L96_force"] = st.number_input('L96_force', value=0.0, step=0.1)
            elif build_args["dyn_sys_opt"] == "KS":
                build_args["KS_system_size"] = st.number_input('KS_system_size', value=5.0, step=0.1)
            if esn_type == "dynsys_pca":
                build_args["pca_components"] = int(st.number_input('pca_components', value=build_args["r_dim"], step=1, min_value=1,
                                                                   max_value=int(build_args["r_dim"]), key="dynsys_pca1"))
                build_args["pca_comps_to_skip"] = int(st.number_input('pca_comps_to_skip', value=0, step=1, min_value=0,
                                                                   max_value=int(build_args["r_dim"])-1, key="dynsys_pca2"))
        elif esn_type == "no_res":
            pass
        elif esn_type == "pca_drop":
            build_args["dims_to_drop"] = st.number_input('dims_to_drop', value=0, step=1)
            if build_args["dims_to_drop"] == 0:
                build_args["dims_to_drop"] = None

        # elif esn_type == "pca_basic":
        #     del build_args["r_to_r_gen_opt"]
        #     build_args["pca_components"] = int(st.number_input('pca_components', value=50, step=1, min_value=1,
        #                                                    max_value=int(build_args["r_dim"])))
        #     build_args["pca_offset"] = st.checkbox("pca_offset", value=1)

    st.markdown("""---""")

    if st.checkbox("custom seed"):
        seed = st.number_input("custom seed", max_value=1000000)
    else:
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
        left, right = st.columns(2)
        with left:
            plot_trajectory_time_series = st.checkbox("Plot Trajectory")
        with right:
            i_dim = st.number_input("i_dim", min_value=0, max_value=x_dim-1)

        if plot_trajectory_time_series:
            fig = plot_original_timeseries(time_series, i_dim, time_boundaries)
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

        esn = build(esn_type, seed, x_dim=x_dim, **build_args)

        w_in_properties_bool = st.checkbox("Show W_in properties:")
        if w_in_properties_bool:
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

        if train_noise:
            x_train = x_train + np.random.randn(*(x_train.shape)) * noise_scale

        esn, x_train_true, x_train_pred, r_train, act_fct_inp_train, r_internal_train, r_input_train, r_gen_train = train(esn,
                                                                                                             x_train,
                                                                                                 t_train_sync)

        r_gen_dim = r_gen_train.shape[1]

        perform_pca_bool = st.checkbox("PCA:")
        if perform_pca_bool:
            pca = perform_pca(r_train, n_components=3)

        show_pca_states_bool = st.checkbox("Show 3-dim Res-PCA:", disabled=not(perform_pca_bool))
        if show_pca_states_bool:
            r_train_pca = pca.transform(r_train)
            fig = plot_pca(r_train_pca)
            st.plotly_chart(fig)

        w_out_properties_bool = st.checkbox("Show W_out properties:")
        if w_out_properties_bool:
            w_out = esn._w_out
            fig = plot_w_out(w_out)
            st.pyplot(fig)

        show_x_train_pred = st.checkbox("Show fitted and real trajectory:")
        if show_x_train_pred:
            figs = plot_train_pred_vs_true_trajectories(x_train_true, x_train_pred)
            for i in range(x_dim):
                st.plotly_chart(figs[i])

        show_x_train_attr = st.checkbox("Show fitted and real attractor:")
        if show_x_train_attr:
            fig = plot_train_pred_vs_true_attractor(x_train_true, x_train_pred)
            st.plotly_chart(fig)

        show_train_error = st.checkbox("Show train error:")
        if show_train_error:
            fig = plot_train_error(x_train_pred, x_train_true)
            st.plotly_chart(fig)

        show_reservoir = st.checkbox("Show Reservoir states: ")
        if show_reservoir:
            fig = show_reservoir_states(r_train)
            st.plotly_chart(fig)

        show_reservoir_histograms_train = st.checkbox("Show Reservoir node value histograms - TRAIN: ")
        if show_reservoir_histograms_train:

            # fig = plot.show_res_state_scatter(r_train,  figsize=(15, 8), sort=True, s=0.1, alpha=0.5)
            # st.pyplot(fig)
            act_fct = esn._act_fct
            fig = plot_reservoir_histograms_train(r_input_train, r_internal_train,
                                    act_fct_inp_train, r_train, act_fct)
            st.pyplot(fig)

        show_poincare_type_map = st.checkbox("Show Poincare Type Map Train:", disabled=disabled)
        if show_poincare_type_map:
            poincare_mode = st.selectbox('Mode', ["maxima", "minima"], key="test")

            fig = plot_poincare_type_map_plotly(x_train_pred, x_train_true, mode=poincare_mode)
            st.plotly_chart(fig)

    st.markdown("""---""")

# Predict with RC:
disabled = False if train_bool else True
with st.expander("Prediction"):
    st.header("Predict with RC: ")
    predict_bool = st.checkbox("Predict with RC", disabled=disabled)
    if predict_bool:
        x_pred = x_pred_list[0]
        # y_pred, y_true, r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred = predict(esn, x_pred, t_pred_sync)
        y_pred, y_true, r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred, r_gen_pred, r_drive = predict_2(esn, x_pred, t_pred_sync)
        show_x_pred = st.checkbox("Show predicted and real trajectory:")
        if show_x_pred:
            figs = plot_prediction_pred_vs_true_trajectories(y_true, y_pred)
            for i in range(x_dim):
                st.plotly_chart(figs[i])

        show_x_pred_attr = st.checkbox("Show predicted and real attractor:")
        if show_x_pred_attr:
            # time_series_dict = {"True": y_true, "Predicted": y_pred}
            # fig = plot.plot_3d_time_series_multiple(time_series_dict)
            fig = plot_prediction_pred_vs_true_attractor(y_true, y_pred)
            st.plotly_chart(fig)

        show_pred_error = st.checkbox("Show predicted error:")
        if show_pred_error:
            fig = plot_pred_error(y_pred, y_true)
            st.plotly_chart(fig)

        left, right = st.columns(2)
        with left:
            valid_times_show = st.checkbox("Show valid_time vs error_threshhold: ")

            if valid_times_show:
                disabled = False
            else:
                disabled = True
        with right:
            if st.checkbox("In lyapunov times", disabled=disabled):
                in_lyapunov_times = {"dt": dt, "LE": lyapunov_exponents[system_option]}
            else:
                in_lyapunov_times = None
        if valid_times_show:
            fig = plot_valid_times_vs_pred_error(y_pred, y_true, in_lyapunov_times=in_lyapunov_times)
            st.plotly_chart(fig)

        # show_r_pred = st.checkbox("Show reservoir states for Prediction:")
        # if show_r_pred:
        #     fig = plot.show_reservoir_states(r_pred)
        #     st.plotly_chart(fig)

        show_pca_states_bool = st.checkbox("Show 3-dim Res-PCA:", disabled=not(perform_pca_bool), key=1)
        if show_pca_states_bool:
            r_pred_pca = pca.transform(r_pred)
            # fig = plot_pca(r_pred_pca)
            r_pca_drive = pca.transform(r_drive)
            fig = plot_pca_with_drive(r_pca_drive, r_pred_pca)
            st.plotly_chart(fig)

        show_reservoir_histograms_pred = st.checkbox("Show Reservoir node value histograms - PRED: ")
        if show_reservoir_histograms_pred:
            act_fct = esn._act_fct
            fig = plot_reservoir_histograms_pred(r_input_pred, r_internal_pred, act_fct_inp_pred, r_pred, act_fct)
            st.pyplot(fig)

        show_trajectory_and_resstates = st.checkbox("Show Reservoir states and trajectories - PRED: ")
        if show_trajectory_and_resstates:
            fig = plot_trajectory_and_resstates(r_input_pred, r_internal_pred, r_pred, y_pred)
            st.pyplot(fig)

        if st.checkbox("Closest point to attractor"):
            fig = plot_closest_distance_to_attractor(y_pred, y_true)
            st.plotly_chart(fig)
            # closest_dist = rescomp.measures.distances_to_closest_point(y_to_test=y_pred, y_true=y_true)
            # st.line_chart(closest_dist)
    st.markdown("""---""")

# Advanced measures:
disabled = False if predict_bool else True
with st.expander("Advanced Measures on Prediction: "):
    st.header("Advanced Measures: ")
    show_correlation_dimension = st.checkbox("Show Correlation Dimension:", disabled=disabled)
    if show_correlation_dimension:
        nr_steps = st.slider("nr_steps", 2., 30., 10.)
        r_min = st.slider("r_min", 0.1, 10., 1.5)
        r_max = st.slider("r_max", 0.5, 10., 5.)
        fig = plot_correlation_dimension(y_pred, y_true, nr_steps, r_min, r_max)
        st.pyplot(fig)

    show_poincare_type_map = st.checkbox("Show Poincare Type Map:", disabled=disabled)
    if show_poincare_type_map:
        left, right = st.columns(2)
        with left:
            poincare_mode = st.selectbox('Mode', ["maxima", "minima"])
        with right:
            value_or_time = st.selectbox('Value or Time', ["value", "time"])
        # fig = plot_poincare_type_map(y_pred, y_true, mode=poincare_mode, figsize=(13, 16))
        # st.pyplot(fig)

        fig = plot_poincare_type_map_plotly(y_pred, y_true, mode=poincare_mode, value_or_time=value_or_time)
        st.plotly_chart(fig)

    show_lyapunov_bool = st.checkbox("Show Lyapunov from Data:")
    if show_lyapunov_bool:
        # Some parameters:
        left, right = st.columns(2)
        with left:
            freq_cut = st.checkbox("freq_cut", value=True)
        with right:
            steps = st.number_input("steps", key=1, min_value=2, max_value=2000, value=100)
        fig = show_lypunov_from_data(y_pred, y_true, dt, freq_cut=freq_cut, steps=steps)
        st.pyplot(fig)
    # disabled = True if normalize else False
    # show_model_likeness = st.checkbox("Show Model Likeness:", disabled=disabled)
    # if show_model_likeness:
    #     # PROTOTYPE:
    #     if system_option == "lorenz":
    #         iterator = lambda x: rescomp.simulations.simulate_trajectory("lorenz", dt, 2, x)[-1]
    #     elif system_option == "roessler":
    #         iterator = lambda x: rescomp.simulations.simulate_trajectory("roessler_sprott", dt, 2, x)[-1]
    #     nr_steps_model_likeness = st.slider("steps", 2, 200, 10)
    #     fig = plot.plot_model_likeness(y_pred, iterator, steps=nr_steps_model_likeness, figsize=(15, 4))
    #     st.pyplot(fig)

disabled = False if predict_bool else True
# Disable also if PCA not active
with st.expander("Free looping of Reservoir Dynamics: "):
    free_loop_with_perturbed_res = st.checkbox("Loop Reservoir with perturbed initial state:")
    if free_loop_with_perturbed_res:
        perturbation_scale = st.number_input("perturbation scale", min_value=0.0, max_value=1000., value=0.005, step=0.001, format="%f")
        perturb_only_pca = st.checkbox("Perturb only along PCA dimension:")
        # time_steps_loop = st.number_input("timesteps for loop", min_value=10, max_value=10000, value=1000)
        # r_pred_previous = r_pred[0, :]
        r_pred_previous = r_drive[-1, :]

        r_dim = r_pred_previous.size
        if perturb_only_pca:
            perturbation_pca = np.random.randn(3)
            perturbation = pca.inverse_transform(perturbation_pca)
        else:
            perturbation = np.random.randn(r_dim)
        pert_len = np.linalg.norm(perturbation)
        if pert_len != 0:
            perturbation = perturbation / pert_len
        perturbation = perturbation * perturbation_scale
        # st.table(perturbation)
        st.table(pd.DataFrame(perturbation).describe())
        r_loop_perturbed = loop_with_perturbation(esn, perturbation, r_pred_previous,
                                                  time_steps_loop=t_pred)
        r_loop_perturbed_pca = pca.transform(r_loop_perturbed)
        r_pred_pca = pca.transform(r_pred)
        fig = plot_pca_with_perturbed_pca(r_loop_perturbed_pca, r_pred_pca)
        st.plotly_chart(fig)

        show_hist = st.checkbox("show value hist")
        if show_hist:
            data = {"real": r_pred_pca[200:, :], "perturbed": r_loop_perturbed_pca[200:, :]}
            fig = plot.show_hist(data, bins=100)
            st.pyplot(fig)

        checkbox = st.checkbox("Attractor Likeness vs. perturbation scale")
        if checkbox:
            # diff = rescomp.measures.perturbation_of_res_dynamics(esn, r_init=r_pred_previous, time_steps=3000, n_ens=30, pert_max=2000, pert_min=0.01)
            # mean_diff = np.mean(diff, axis=0)
            # # std_diff = np.std(diff, axis=0)
            # df = pd.DataFrame()
            #
            # fig = px.line(df, x=sweep_variable, y="valid times", error_y="valid times std", color="Other Parameters",
            #               width=figsize[0],
            #               height=figsize[1], )
            # fig = plt.figure()
            # plt.plot(mean_diff)
            # st.pyplot(fig)

            left, right = st.columns(2)
            with left:
                time_steps = st.number_input("time_steps", value=1000, min_value=200, key="timesteps_pert")
            with right:
                n_ens = st.number_input("n_ens", value=10, min_value=1, key="nens_pert")

            l, m, r = st.columns(3)
            with l:
                pert_min = st.number_input("pert_scale_min", value=0.0, key="minpert", format="%f")
            with m:
                pert_max = st.number_input("pert_scale_max", value=5., key="maxpert", format="%f")
            with r:
                pert_steps = st.number_input("pert_steps", value=5, key="pertsteps")
            df, fig1, fig2 = plot_pde_difference_vs_pert(esn, r_init=r_pred_previous, time_steps=time_steps, n_ens=n_ens, pert_max=pert_max,
                                              pert_min=pert_min, pert_steps=pert_steps)
            # df = rescomp.measures.perturbation_of_res_dynamics(esn, r_init=r_pred_previous, time_steps=2000, n_ens=30, pert_max=2, pert_min=0.01, pert_steps=20)
            # # fig = px.line(df, x="pert_scale", y="diff_median", error_y="diff_std")
            # fig = px.line(df, x="pert_scale", y="diff_median", error_y_minus="diff_lower_quartile", error_y="diff_upper_quartile")
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.table(df)

with st.expander("Visualization of arbitrary PCA components: "):

    if st.checkbox("perform pca for all components"):

        train_predict = st.selectbox("train or predict", ["train", "predict"])
        if train_predict == "train":
            r_use = r_train
        elif train_predict == "predict":
            r_use = r_pred

        r_dim = int(build_args["r_dim"])

        pca_all = perform_pca(r_use, n_components=r_dim)
        l, m, r = st.columns(3)
        with l:
            i_x = st.number_input("pca-comp for x-dim", min_value=0, max_value=r_dim-1, value=0)
        with m:
            i_y = st.number_input("pca-comp for y-dim", min_value=0, max_value=r_dim-1, value=1)
        with r:
            i_z = st.number_input("pca-comp for z-dim", min_value=0, max_value=r_dim-1, value=2)

        mode = st.selectbox("plot mode", ["line", "scatter"])

        r_use_transform = pca_all.transform(r_use)[:, [i_x, i_y, i_z]]
        fig = plot_pca(r_use_transform, mode=mode)
        st.plotly_chart(fig)

        explained_var = pca_all.explained_variance_ratio_
        st.write("Explained variance of pca components: ")
        st.line_chart(explained_var)

        explained_var_cumulative = np.cumsum(explained_var)
        st.write("Explained variance cumulative of pca components: ")
        st.line_chart(explained_var_cumulative)


with st.expander("Visualization of W_out magnitudes: "):
    if st.checkbox("Plot W_out magnitudes and r_gen std"):
        w_out = esn._w_out

        w_out_norm = w_out.flatten().dot(w_out.flatten())
        st.write(f"|W|^2 = {np.round(w_out_norm, 1)}")

        r_gen_data = {"r_gen_train": r_gen_train, "r_gen_pred": r_gen_pred}

        log_y = st.checkbox("log_y")

        figs = plot_w_out_and_r_gen_std_quantites(r_gen_data, w_out, log_y)
        for fig in figs:
            st.plotly_chart(fig)
        #
        # fig1, fig2 = plot_w_out_and_r_gen_std(w_out, r_gen_train, r_gen_pred)
        #
        # fig = plot.plot_wout_magnitudes(w_out, figsize=(650, 500))
        # st.plotly_chart(fig)
        # fig = plot.plot_state_std({"r_gen_train": r_gen_train, "r_gen_pred": r_gen_pred}, figsize=(650, 500),
        #                           title="std of r_gen during train and pred")
        # st.plotly_chart(fig)
        #
        # df, figs = plot.plot_w_out_times_r_gen_state_std({"r_gen_train": r_gen_train, "r_gen_pred": r_gen_pred}, w_out)
        # st.dataframe(df)

        if st.checkbox("R_gen value histograms"):
            steps = st.number_input("time_steps for hist", value=100)
            fig = plot.plot_histogram(r_gen_data, steps=steps, log_y=log_y)
            st.plotly_chart(fig)

        if st.checkbox("Meaning of r_gen components for prediction"):
            left, right = st.columns(2)
            with left:
                train_predict = st.selectbox("train or predict", ["train", "predict"], key="tp2")
            with right:
                mode = st.selectbox("plotting mode", ["line", "scatter"])
            st.write("r_gen components to keep exclude/include x to y")

            left, mid, right = st.columns(3)
            with left:
                x = st.number_input("x", value=int(r_gen_dim/2), min_value=0,
                                max_value=r_gen_dim-1, key="x1")
            with mid:
                y = st.number_input("y", value=r_gen_dim-2, min_value=0,
                                max_value=r_gen_dim-1, key="y1")
            with right:
                exclude_or_include = st.selectbox("exclude or include", ["exclude", "include"])

            if train_predict == "train":
                r_gen = r_gen_data["r_gen_train"]
                out_real = x_train_pred
            elif train_predict == "predict":
                r_gen = r_gen_data["r_gen_pred"]
                out_real = y_pred

            if exclude_or_include == "include":
                out = w_out[:, x:y+1] @ (r_gen[:, x:y+1]).T
            elif exclude_or_include == "exclude":
                w_out = np.delete(w_out, np.s_[x:y+1], axis=1)
                r_gen = np.delete(r_gen, np.s_[x:y+1], axis=1)
                out = w_out @ r_gen.T
            # print(w_out.shape, r_gen.shape)
            data = {"components removed": out.T, "real": out_real}

            fig = plot.plot_3d_time_series_multiple(data, mode=mode, size=1)
            st.plotly_chart(fig)

        if st.checkbox("show r_gen reservoir states: "):
            left, right = st.columns(2)
            with left:
                train_predict = st.selectbox("train or predict", ["train", "predict"], key="tp3")
            with right:
                log_y = st.checkbox("log_y", key="log_y2")
            if train_predict == "train":
                r_gen = r_gen_data["r_gen_train"]
            elif train_predict == "predict":
                r_gen = r_gen_data["r_gen_pred"]

            if log_y:
                r_gen = np.log(r_gen)
            fig = show_reservoir_states(r_gen)
            st.plotly_chart(fig)
