"""Python file that includes streamlit elements that are used to build/train/predict with an esn."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import streamlit as st

import rescomp
import rescomp.esn_new_update_code as esn
from streamlit_project.app_fragments import streamlit_utilities as utils


def esn_hash(obj):
    items = sorted(obj.__dict__.items(), key=lambda it: it[0])
    return hash((type(obj),) + tuple(items))


ESN_DICT = {"ESN_normal": esn.ESN_normal,
            "ESN_pca": esn.ESN_pca,
            "ESN_pca_after_rgen": esn.ESN_pca_after_rgen,
            "ESN_normal_centered": esn.ESN_normal_centered,
            # "ESN_factor_analysis": esn.ESN_factor_analysis,
            # "ESN_fast_ica": esn.ESN_fast_ica,
            "ESN_output_hybrid": esn.ESN_output_hybrid,
            "ESN_strong": esn.ESN_strong
            }

# ESN_HASH_FUNCS = {val: hash for val in ESN_DICT.values()}

ESN_HASH_FUNC = {esn._ResCompCore: esn_hash}

W_IN_TYPES = ["random_sparse", "ordered_sparse", "random_dense_uniform", "random_dense_gaussian"]
BIAS_TYPES = ["no_bias", "random_bias", "constant_bias"]
NETWORK_TYPES = ["erdos_renyi", "scale_free", "small_world", "random_directed",
                 "random_dense_gaussian",
                 "scipy_sparse"]
ACTIVATION_FUNCTIONS = ["tanh", "sigmoid", "relu", "linear"]
R_TO_R_GEN_TYPES = ["linear_r", "linear_and_square_r", "output_bias", "bias_and_square_r",
                    "linear_and_square_r_alt",
                    "exponential_r", "bias_and_exponential_r"]

ESN_TYPING = Any


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def build(esn_type: str, seed: int, x_dim: int, build_args: dict[str, Any]) -> ESN_TYPING:
    """Build the esn class.

    Args:
        esn_type: One of the esn types defined in ESN_DICT.
        seed: Set the global seed. TODO: maybe dont set global seed?
        x_dim: The x_dimension of the data to be predicted.
        build_args: The build args parsed to esn_obj.build.

    Returns:
        The built esn.
    """
    if esn_type in ESN_DICT.keys():
        esn = ESN_DICT[esn_type]()
    else:
        raise Exception("This esn_type is not accounted for")

    seed_args = _get_seed_args_in_build(esn)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000000, len(seed_args))
    for i_seed, seed_arg in enumerate(seed_args):
        build_args[seed_arg] = seeds[i_seed]

    build_kwargs = rescomp.utilities._remove_invalid_args(esn.build, build_args)

    esn.build(x_dim, **build_kwargs)
    return esn


def _get_seed_args_in_build(esn_obj: ESN_TYPING) -> list[str, ...]:
    """Utility function to get all the seed kwargs in the esn.build function.

    Args:
        esn_obj: The esn object with the build method.

    Returns:
        List of keyword argument names of build, that have "seed" in it.
    """
    build_func = esn_obj.build
    args = inspect.signature(build_func).parameters
    return [arg_name for arg_name in args.keys() if "seed" in arg_name]


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def build_with_seed(esn_type: str, seed: int, x_dim: int, **kwargs) -> ESN_TYPING:
    """Build the esn class.
    # TODO: This is more or less for pca_esn_demo app! But the seeding is also good for normal build.

    Args:
        esn_type: One of the esn types defined in ESN_DICT.
        seed: Set the seed to create the other seeds.
        x_dim: The x_dimension of the data to be predicted.
        **kwargs: All other build args.

    Returns:
        The build esn.
    """
    if esn_type in ESN_DICT.keys():
        esn = ESN_DICT[esn_type]()
    else:
        raise Exception("This esn_type is not accounted for")

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000000, 5)
    kwargs["input_noise_seed"] = seeds[0]
    kwargs["r_train_noise_seed"] = seeds[1]
    kwargs["network_seed"] = seeds[2]
    kwargs["bias_seed"] = seeds[3]
    kwargs["w_in_seed"] = seeds[4]

    build_kwargs = rescomp.utilities._remove_invalid_args(esn.build, kwargs)

    with utils.temp_seed(seed):
        esn.build(x_dim, **build_kwargs)
    return esn


def st_esn_strong_args(key: str | None = None) -> dict[str, object]:
    """Streamlit elements to specify the additional settings of esn_strong.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A dictionary containing the esn_strong build args.
    """
    esn_strong_build_args = {}
    esn_strong_build_args["perform_pca_bool"] = st.checkbox('perform_pca_bool',
                                                            value=False,
                                                            key=f"{key}__st_esn_strong_build_args__pca")

    esn_strong_build_args["w_out_from_w_out_pca_bool"] = st.checkbox('w_out_from_w_out_pca_bool',
                                                                     value=False,
                                                                     key=f"{key}__st_esn_strong_build_args__pcawoutback")

    esn_strong_build_args["n_pcr_comps"] = int(st.number_input('n_pcr_comps',
                                                               value=500,
                                                               step=1,
                                                               key=f"{key}__st_esn_strong_build_args__n_pcr_comps"))

    esn_strong_build_args["input_noise_scale"] = st.number_input('input_noise_scale',
                                                                 value=0.0,
                                                                 step=0.01,
                                                                 min_value=0.0,
                                                                 key=f"{key}__st_esn_strong_build_args__in_noise",
                                                                 format="%f")

    esn_strong_build_args["r_train_noise_scale"] = st.number_input('r_train_noise_scale',
                                                                   value=0.0,
                                                                   step=0.01,
                                                                   min_value=0.0,
                                                                   key=f"{key}__st_esn_strong_build_args__r_noise",
                                                                   format="%f")
    return esn_strong_build_args


def st_select_esn_type(esn_sub_section: tuple[str, ...] | None = None,
                       key: str | None = None) -> str:
    """Streamlit elements to specify the esn type.

    Args:
        esn_sub_section: A subsection of the keys in ESN_DICT, or if None, take all of ESN_DICT.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The esn_type as a string.
    """
    if esn_sub_section is None:
        esn_dict = ESN_DICT
    else:
        esn_dict = {esn_name: esn_class for esn_name, esn_class in ESN_DICT.items()
                    if esn_name in esn_sub_section}
        if len(esn_dict) == 0:  # TODO: proper error
            raise Exception(f"The systems in {esn_sub_section} are not accounted for.")

    esn_type = st.selectbox('esn type', esn_dict.keys(), key=f"{key}__st_select_esn_type")
    return esn_type


def st_basic_esn_build(key: str | None = None) -> dict[str, Any]:
    """Streamlit elements to specify the basic esn settings.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The basic esn build_args as a dictionary.
    """

    basic_build_args = {}
    basic_build_args["r_dim"] = int(st.number_input('Reservoir Dim', value=500, step=1,
                                                    key=f"{key}__st_basic_esn_build__rd"))
    basic_build_args["r_to_r_gen_opt"] = st.selectbox('r_to_r_gen_opt', R_TO_R_GEN_TYPES,
                                                      key=f"{key}__st_basic_esn_build__rrgen")
    basic_build_args["act_fct_opt"] = st.selectbox('act_fct_opt', ACTIVATION_FUNCTIONS,
                                                   key=f"{key}__st_basic_esn_build__actfct")
    basic_build_args["node_bias_opt"] = st.selectbox('node_bias_opt', BIAS_TYPES,
                                                     key=f"{key}__st_basic_esn_build__nbo")
    disabled = True if basic_build_args["node_bias_opt"] == "no_bias" else False
    basic_build_args["bias_scale"] = st.number_input('bias_scale', value=0.1, step=0.1,
                                                     disabled=disabled,
                                                     key=f"{key}__st_basic_esn_build__bs")
    basic_build_args["leak_factor"] = st.number_input('leak_factor', value=0.0, step=0.01,
                                                      min_value=0.0, max_value=1.0,
                                                      key=f"{key}__st_basic_esn_build__lf")
    basic_build_args["w_in_opt"] = st.selectbox('w_in_opt', W_IN_TYPES,
                                                key=f"{key}__st_basic_esn_build__winopt")
    basic_build_args["w_in_scale"] = st.number_input('w_in_scale', value=1.0, step=0.1,
                                                     key=f"{key}__st_basic_esn_build__winsc")
    basic_build_args["input_noise_scale"] = st.number_input('input noise scale',
                                                            value=0.0,
                                                            format="%f",
                                                            key=f"{key}__st_basic_esn_build__inpnoisescale")
    log_reg_param = st.number_input('Log regulation parameter', value=-7., step=1., format="%f",
                                    key=f"{key}__st_basic_esn_build__reg")
    basic_build_args["reg_param"] = 10 ** (log_reg_param)

    return basic_build_args


def st_network_build_args(key: str | None = None) -> dict[str, object]:
    """Streamlit elements to specify the network settings.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A dictionary containing the network build args.
    """
    network_build_args = {}
    network_build_args["n_rad"] = st.number_input('n_rad', value=0.1, step=0.1, format="%f",
                                                  key=f"{key}__st_network_build_args__nrad")
    network_build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=5.0, step=0.1,
                                                      key=f"{key}__st_network_build_args__ndeg")
    network_build_args["n_type_opt"] = st.selectbox('n_type_opt', NETWORK_TYPES,
                                                    key=f"{key}__st_network_build_args__nopt")
    return network_build_args


def st_pca_build_args(r_dim: int,
                      key: str | None = None) -> dict[str, object]:
    """Streamlit elements to specify the Settings for ESN_pca.

    Args:
        r_dim: The reservoir dimension, as the maximum value of the principal components.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A dictionary containing the pca esn build args.
    """

    pca_build_args = {}
    pca_build_args["pca_components"] = int(
        st.number_input('principal components',
                        value=r_dim,
                        step=1,
                        min_value=1,
                        max_value=int(r_dim),
                        key=f"{key}__st_pca_build_args__pc")
    )
    return pca_build_args


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def train_return_res(esn_obj: ESN_TYPING, x_train: np.ndarray, t_train_sync: int,
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], ESN_TYPING]:
    """Train the esn_obj with a given x_train and t_train-sync and return internal reservoir states.

    TODO: check when to use this func and when to use train.

    Args:
        esn_obj: The esn_obj, that has a train method.
        x_train: The np.ndarray of shape (t_train_sync_steps + t_train_steps, sys_dim)
        t_train_sync: The number of time steps used for syncing the esn before training.

    Returns:
        Tuple with the fitted output, the real output and reservoir dictionary containing states
        for r_act_fct_inp, r_internal, r_input, r, r_gen, and the esn_obj.
    """
    esn_obj.train(x_train,
                  sync_steps=t_train_sync,
                  save_y_train=True,
                  save_out=True,
                  save_res_inp=True,
                  save_r_internal=True,
                  save_r=True,
                  save_r_gen=True
                  )

    y_train_true = esn_obj.get_y_train()
    y_train_fit = esn_obj.get_out()

    res_state_dict = {}
    res_state_dict["r_act_fct_inp"] = esn_obj.get_act_fct_inp()
    res_state_dict["r_internal"] = esn_obj.get_r_internal()
    res_state_dict["r_input"] = esn_obj.get_res_inp()
    res_state_dict["r"] = esn_obj.get_r()
    res_state_dict["r_gen"] = esn_obj.get_r_gen()

    return y_train_fit, y_train_true, res_state_dict, esn_obj


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def predict_return_res(esn_obj: ESN_TYPING, x_pred: np.ndarray, t_pred_sync: int
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], ESN_TYPING]:
    """Predict with the esn_obj with a given x_pred and x_pred_sync and return internal reservoir states.

    TODO: check when to use this func and when to use predict.

    Args:
        esn_obj: The esn_obj, that has a predict method.500
        x_pred: The np.ndarray of shape (t_pred_sync_steps + t_pred_steps, sys_dim)
        t_pred_sync: The number of time steps used for syncing the esn before prediction.

    Returns:
        Tuple with the fitted output, the real output and reservoir dictionary containing states
        for r_act_fct_inp, r_internal, r_input, r, r_gen, and the esn_obj.
    """
    y_pred, y_pred_true = esn_obj.predict(x_pred,
                                          sync_steps=t_pred_sync,
                                          save_res_inp=True,
                                          save_r_internal=True,
                                          save_r=True,
                                          save_r_gen=True
                                          )
    res_state_dict = {}
    res_state_dict["r_act_fct_inp"] = esn_obj.get_act_fct_inp()
    res_state_dict["r_internal"] = esn_obj.get_r_internal()
    res_state_dict["r_input"] = esn_obj.get_res_inp()
    res_state_dict["r"] = esn_obj.get_r()
    res_state_dict["r_gen"] = esn_obj.get_r_gen()

    return y_pred, y_pred_true, res_state_dict, esn_obj
