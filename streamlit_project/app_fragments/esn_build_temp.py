from __future__ import annotations

from typing import Any, Callable

import streamlit as st
import numpy as np

import rescomp
import rescomp.esn_new_update_code as esn_new
import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.app_fragments.utilities_temp as util

ESN_DICT = {"ESN_normal": esn_new.ESN_normal,
            "ESN_output_hybrid": esn_new.ESN_output_hybrid,
            "ESN_input_hybrid": esn_new.ESN_input_hybrid,
            "ESN_full_hybrid": esn_new.ESN_full_hybrid,
            "ESN_full_hybrid_same": esn_new.ESN_full_hybrid_same,
            "ESN_output_hybrid_preprocess": esn_new.ESN_output_hybrid_preprocess,
            "ESN_outp_var_preproc": esn_new.ESN_outp_var_preproc,
            }

ESN_HASH_FUNCS = {val: hash for val in ESN_DICT.values()}

# ESN_HASH_FUNCS = {esn_new.ESN_normal: hash,
#                   esn_new.ESN_dynsys: hash,
#                   esn_new.ESN_difference: hash,
#                   esn_new.ESN_output_hybrid: hash,
#                   esn_new.ESN_no_res: hash,
#                   esn_new.ESN_pca_adv: hash,
#                   esn_new.ESN_pca: hash,
#                   esn_new.ESN_dynsys_pca: hash,
#                   esn_new.ESN_normal_centered: hash,
#                   esn_new.ESN_pca_noise: hash,
#                   esn_new.ESN_output_hybrid_pca: hash,
#                   esn_new.ESN_input_hybrid: hash,
#                   esn_new.ESN_input_hybrid_pca: hash,
#                   esn_new.ESN_full_hybrid: hash,
#                   esn_new.ESN_full_hybrid_pca: hash,
#                   }


esn_types = ["normal", "dynsys", "difference", "no_res", "pca", "dynsys_pca", "normal_centered", "model_and_pca",
             "pca_noise", "input_to_rgen", "pca_drop", "output_hybrid", "output_hybrid_pca", "input_hybrid", "input_hybrid_pca",
             "full_hybrid", "full_hybrid_pca"]
w_in_types = ["random_sparse", "ordered_sparse", "random_dense_uniform", "random_dense_gaussian"]
bias_types = ["no_bias", "random_bias", "constant_bias"]
network_types = ["erdos_renyi", "scale_free", "small_world", "random_directed", "random_dense_gaussian",
                 "scipy_sparse"]
activation_functions = ["tanh", "sigmoid", "relu", "linear"]
r_to_r_gen_types = ["linear_r", "linear_and_square_r", "output_bias", "bias_and_square_r", "linear_and_square_r_alt",
                    "exponential_r", "bias_and_exponential_r"]
dyn_sys_types = ["L96", "KS"]


@st.experimental_memo
def get_random_int():
    print("Get new seed")
    return np.random.randint(1, 1000000)


# @st.experimental_singleton #TODO: doesnt work for hybrid
@st.cache(hash_funcs=ESN_HASH_FUNCS)
def build(esntype, seed, x_dim=3, **kwargs):
    # if esntype == "normal":
    #     esn = esn_new.ESN_normal()
    # elif esntype == "dynsys":
    #     esn = esn_new.ESN_dynsys()
    # elif esntype == "difference":
    #     esn = esn_new.ESN_difference()
    # elif esntype == "no_res":
    #     esn = esn_new.ESN_no_res()
    # elif esntype == "pca_drop":
    #     esn = esn_new.ESN_pca_adv()
    # elif esntype == "input_to_rgen":
    #     esn = esn_new.ESN_output_hybrid()  # but dont give a model -> i.e. its just the identiy: f:x -> x
    # elif esntype == "pca":
    #     esn = esn_new.ESN_pca()
    # elif esntype == "dynsys_pca":
    #     esn = esn_new.ESN_dynsys_pca()
    # elif esntype == "normal_centered":
    #     esn = esn_new.ESN_normal_centered()
    # elif esntype == "pca_noise":
    #     esn = esn_new.ESN_pca_noise()
    # elif esntype == "model_and_pca":
    #     esn = esn_new.ESN_output_hybrid_pca()
    # elif esntype == "output_hybrid":
    #     esn = esn_new.ESN_output_hybrid()
    # elif esntype == "output_hybrid_pca":
    #     esn = esn_new.ESN_output_hybrid_pca()
    # elif esntype == "input_hybrid":
    #     esn = esn_new.ESN_input_hybrid()
    # elif esntype == "input_hybrid_pca":
    #     esn = esn_new.ESN_input_hybrid_pca()
    # elif esntype == "full_hybrid":
    #     esn = esn_new.ESN_full_hybrid()
    # elif esntype == "full_hybrid_pca":
    #     esn = esn_new.ESN_full_hybrid_pca()
    if esntype in ESN_DICT.keys():
        esn = ESN_DICT[esntype]()
    else:
        raise Exception("This esntype is not accounted for")
    np.random.seed(seed)

    build_kwargs = rescomp.utilities._remove_invalid_args(esn.build, kwargs)
    esn.build(x_dim, **build_kwargs)
    return esn


@st.cache(hash_funcs=ESN_HASH_FUNCS)
def train(esn, x_train, t_train_sync):
    print("train")
    esn.train(x_train, sync_steps=t_train_sync, save_res_inp=False, save_r_internal=False, save_r=False,
              save_r_gen=False, save_out=True, save_y_train=True)

    esn.train(x_train, sync_steps=t_train_sync, save_res_inp=True, save_r_internal=True, save_r=True,
              save_r_gen=False, save_out=True, save_y_train=True)

    # x_train_true = x_train[1+t_train_sync:]
    act_fct_inp_train = esn.get_act_fct_inp()
    r_internal_train = esn.get_r_internal()
    r_input_train = esn.get_res_inp()
    r_train = esn.get_r()
    x_train_true = esn.get_y_train()
    # r_gen_train = esn.get_r_gen()
    x_train_pred = esn.get_out()
    print("shapes: ", x_train_true.shape, x_train_pred.shape)
    # return esn, x_train_true, x_train_pred #, r_train, act_fct_inp_train, r_internal_train, r_input_train, r_gen_train
    return esn, x_train_true, x_train_pred, r_train, act_fct_inp_train, r_internal_train, r_input_train


@st.cache(hash_funcs=ESN_HASH_FUNCS)
def predict(esn, x_pred, t_pred_sync):
    esn.reset_r()

    if t_pred_sync > 0:
        sync = x_pred[:t_pred_sync]
        true_data = x_pred[t_pred_sync:]
        esn.drive(sync, save_r=True)
    else:
        true_data = x_pred

    r_drive = esn.get_r()

    steps = true_data.shape[0]
    y_pred, y_true = esn.loop(steps, save_res_inp=False, save_r_internal=False, save_r=False,
                              save_r_gen=False), \
                     true_data

    # r_pred = esn.get_r()
    # r_gen_pred = esn.get_r_gen()
    # act_fct_inp_pred = esn.get_act_fct_inp()
    # r_internal_pred = esn.get_r_internal()
    # r_input_pred = esn.get_res_inp()

    return y_pred, y_true,  # r_pred, act_fct_inp_pred, r_internal_pred, r_input_pred, r_gen_pred, r_drive


def st_basic_esn_build(esn_sub_section: tuple[str, ...] | None = None):

    if esn_sub_section is None:
        esn_dict = ESN_DICT
    else:
        esn_dict = {esn_name: esn_class for esn_name, esn_class in ESN_DICT.items()
                       if esn_name in esn_sub_section}
        if len(esn_dict) == 0:  # TODO: proper error
            raise Exception(f"The systems in {esn_sub_section} are not accounted for.")

    # TODO: also a dict with all the esn types:
    # esn_type = st.selectbox('esn type', esn_types)
    esn_type = st.selectbox('esn type', esn_dict.keys())

    basic_build_args = {}
    basic_build_args["r_dim"] = st.number_input('Reservoir Dim', value=500, step=1)
    basic_build_args["r_to_r_gen_opt"] = st.selectbox('r_to_r_gen_opt', r_to_r_gen_types)
    basic_build_args["act_fct_opt"] = st.selectbox('act_fct_opt', activation_functions)
    basic_build_args["node_bias_opt"] = st.selectbox('node_bias_opt', bias_types)
    disabled = True if basic_build_args["node_bias_opt"] == "no_bias" else False
    basic_build_args["bias_scale"] = st.number_input('bias_scale', value=0.1, step=0.1,
                                                     disabled=disabled)
    basic_build_args["leak_factor"] = st.number_input('leak_factor', value=0.0, step=0.01,
                                                      min_value=0.0, max_value=1.0)
    basic_build_args["w_in_opt"] = st.selectbox('w_in_opt', w_in_types)
    basic_build_args["w_in_scale"] = st.number_input('w_in_scale', value=1.0, step=0.1)
    log_reg_param = st.number_input('Log regulation parameter', value=-7., step=1., format="%f")
    basic_build_args["reg_param"] = 10 ** (log_reg_param)

    return esn_type, basic_build_args


def st_network_build_args():
    network_build_args = {}
    network_build_args["n_rad"] = st.number_input('n_rad', value=0.4, step=0.1, format="%f")
    network_build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=3.0, step=0.1)
    network_build_args["n_type_opt"] = st.selectbox('n_type_opt', network_types)
    return network_build_args


def st_hybrid_build_args(esn_type: str, system_name: str, system_parameters: dict[str, Any]
                         ) -> dict[str, Any]:
    hybrid_build_args = {}

    if esn_type == "ESN_full_hybrid":
        same_model_bool = st.checkbox("Same model", value=True)
        if same_model_bool:
            model, hybrid_build_args["Input Model Parameters"] = syssim.st_get_model_system(system_name, system_parameters)
            hybrid_build_args["input_model"] = model
            hybrid_build_args["output_model"] = model
        else:
            st.write("Input Model: ")
            hybrid_build_args["input_model"], hybrid_build_args["Input Model Parameters"] = syssim.st_get_model_system(system_name, system_parameters, unique_key="input")
            util.line()
            st.write("Output Model: ")
            hybrid_build_args["output_model"], hybrid_build_args["Output Model Parameters"] = syssim.st_get_model_system(system_name, system_parameters, unique_key="output")

    if esn_type in ["ESN_full_hybrid", "ESN_input_hybrid", "ESN_full_hybrid_same"]:
        util.line()
        hybrid_build_args["model_to_network_factor"] = st.number_input("model_to_network_factor",
                                                                value=0.5, min_value=0.0,
                                                                max_value=1.0)
    if esn_type in ["ESN_input_hybrid", "ESN_full_hybrid_same"]:
        hybrid_build_args["input_model"], hybrid_build_args["Input Model Parameters"] = syssim.st_get_model_system(system_name, system_parameters)
    if esn_type in ["ESN_output_hybrid", "ESN_output_hybrid_preprocess"]:
        hybrid_build_args["output_model"], hybrid_build_args["Output Model Parameters"] = syssim.st_get_model_system(system_name,
                                                                      system_parameters)
    if "preprocess" in esn_type:
        hybrid_build_args["noise_scale"] = st.number_input("train noise scale", value=0.0, step=0.01,
                                                        format="%f")

    util.line()
    for key, val in hybrid_build_args.items():
        if "Parameters" in key:
            st.write(f"{key}: ")
            for key2, val2 in val.items():
                st.write(f"{key2} = {val2}")

    return hybrid_build_args


def st_build_esn(system_name, system_parameters):
    # TODO: slowly build up build_args with many steps.
    esn_type = st.selectbox('esn type', esn_types)

    build_args = {}
    build_args["r_dim"] = st.number_input('Reservoir Dim', value=1000, step=1)
    build_args["r_to_r_gen_opt"] = st.selectbox('r_to_r_gen_opt', r_to_r_gen_types)
    build_args["act_fct_opt"] = st.selectbox('act_fct_opt', activation_functions)
    build_args["node_bias_opt"] = st.selectbox('node_bias_opt', bias_types)
    disabled = True if build_args["node_bias_opt"] == "no_bias" else False
    build_args["bias_scale"] = st.number_input('bias_scale', value=0.1, step=0.1,
                                               disabled=disabled)  # disable if needed
    build_args["leak_factor"] = st.number_input('leak_factor', value=0.0, step=0.01, min_value=0.0, max_value=1.0)
    build_args["w_in_opt"] = st.selectbox('w_in_opt', w_in_types)
    build_args["w_in_scale"] = st.number_input('w_in_scale', value=1.0, step=0.1)
    log_reg_param = st.number_input('Log regulation parameter', value=-7., step=1., format="%f")
    build_args["reg_param"] = 10 ** (log_reg_param)

    # settings depending on esn_type:
    with st.expander("ESN type specific settings:"):
        if esn_type in ["normal", "difference", "input_to_rgen", "pca", "normal_centered", "pca_noise",
                        "model_and_pca", "output_hybrid", "output_hybrid_pca", "input_hybrid", "input_hybrid_pca",
                            "full_hybrid", "full_hybrid_pca"]:
            # network:
            build_args["n_rad"] = st.number_input('n_rad', value=0.4, step=0.1, format="%f")
            build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=3.0, step=0.1)
            build_args["n_type_opt"] = st.selectbox('n_type_opt', network_types)
            if esn_type == "difference":
                build_args["dt_difference"] = st.number_input('dt_difference', value=0.1, step=0.01)
            if esn_type in ["pca", "pca_noise", "model_and_pca", "output_hybrid_pca", "input_hybrid_pca", "full_hybrid_pca"]:
                build_args["pca_components"] = int(
                    st.number_input('pca_components', value=build_args["r_dim"], step=1, min_value=1,
                                    max_value=int(build_args["r_dim"]), key="pca2"))
                build_args["pca_comps_to_skip"] = int(st.number_input('pca_comps_to_skip', value=0, step=1, min_value=0,
                                                                      max_value=int(build_args["r_dim"]) - 1))
                left, right = st.columns(2)
                with left:
                    build_args["norm_with_expl_var"] = st.checkbox("norm with explained var", value=False)
                with right:
                    build_args["centering_pre_trans"] = st.checkbox("center data before transformation", value=True)

                st.write("EXperimental pca settings: ")
                if st.checkbox("mix first pca components. (sets r_to_r_gen_opt to CUSTOM)"):
                    r_gen_custom = lambda r: np.array([r[0]**2, r[1]**2, r[2]**2, r[0]*r[1], r[0]*r[1], r[1]*r[2]])
                    build_args["r_to_r_gen_opt"] = lambda r, x: np.hstack((np.hstack((r,r_gen_custom(r))), 1))

                if esn_type == "pca_noise":
                    build_args["train_noise_scale"] = st.number_input('train noise scale', value=0.01, step=0.1,
                                                                      min_value=0.0, format="%f")

                    build_args["noise_option"] = st.selectbox('noise option', ["pre_r_gen", "post_r_gen"])

            if esn_type in ["output_hybrid", "output_hybrid_pca", "input_hybrid", "input_hybrid_pca",
                            "full_hybrid", "full_hybrid_pca"]:
                pass
                # model = syssim.st_get_model_system(system_name, system_parameters)


            #     st.warning("Works only for lorenz and Thomas system for now")
            #     if system_option == "lorenz":
            #         eps = st.number_input("change rho", value=0.01, min_value=0.0)
            #         modified_parameters = {"sigma": 10, "rho": 28 * (1 + eps), "beta": 8/3}
            #         model_fct = lambda x: rescomp.simulations.simulate_trajectory(sys_flag=system_option, dt=dt,
            #                                                                                 time_steps=2, starting_point=x,
            #                                                       **modified_parameters)[-1, :]
            #     elif system_option == "thomas":
            #         eps = st.number_input("change b", value=0.01, min_value=0.0)
            #         modified_parameters = {"b": 0.18* (1 + eps)}
            #         model_fct = lambda x: rescomp.simulations.simulate_trajectory(sys_flag=system_option, dt=dt,
            #                                                                                 time_steps=2, starting_point=x,
            #                                                       **modified_parameters)[-1, :]
            #
            #     elif system_option == "roessler_sprott":
            #         eps = st.number_input("change c", value=0.01, min_value=0.0)
            #         modified_parameters = {"a": 0.2, "b": 0.2, "c": 5.7*(1+eps)}
            #         model_fct = lambda x: rescomp.simulations.simulate_trajectory(sys_flag=system_option, dt=dt,
            #                                                                                 time_steps=2, starting_point=x,
            #                                                       **modified_parameters)[-1, :]
            #
            #     else:
            #         st.warning("This system is not supported!")
            #         model_fct = lambda x: x
            #     if esn_type in ["output_hybrid", "output_hybrid_pca", "full_hybrid", "full_hybrid_pca"]:
            #         build_args["output_model"] = model_fct
            #     if esn_type in ["input_hybrid", "input_hybrid_pca", "full_hybrid", "full_hybrid_pca"]:
            #         build_args["input_model"] = model_fct
            #         build_args["model_to_network_factor"] = st.number_input("model_to_network_factor", value=0.5,
            #                                                                 min_value=0.0, max_value=1.0)

        elif esn_type in ["dynsys", "dynsys_pca"]:
            build_args["dyn_sys_opt"] = st.selectbox('dyn_sys_opt', dyn_sys_types)
            build_args["dyn_sys_dt"] = st.number_input('dyn_sys_dt', value=0.1, step=0.01)
            build_args["scale_factor"] = st.number_input('scale_factor', value=1.0, step=0.1)
            if build_args["dyn_sys_opt"] == "L96":
                build_args["L96_force"] = st.number_input('L96_force', value=0.0, step=0.1)
            elif build_args["dyn_sys_opt"] == "KS":
                build_args["KS_system_size"] = st.number_input('KS_system_size', value=5.0, step=0.1)
            if esn_type == "dynsys_pca":
                build_args["pca_components"] = int(
                    st.number_input('pca_components', value=build_args["r_dim"], step=1, min_value=1,
                                    max_value=int(build_args["r_dim"]), key="dynsys_pca1"))
                build_args["pca_comps_to_skip"] = int(st.number_input('pca_comps_to_skip', value=0, step=1, min_value=0,
                                                                      max_value=int(build_args["r_dim"]) - 1,
                                                                      key="dynsys_pca2"))
        elif esn_type == "no_res":
            pass
        elif esn_type == "pca_drop":
            build_args["dims_to_drop"] = st.number_input('dims_to_drop', value=0, step=1)
            if build_args["dims_to_drop"] == 0:
                build_args["dims_to_drop"] = None
    return esn_type, build_args


if __name__ == '__main__':
    pass
