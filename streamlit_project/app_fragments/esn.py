"""Python file that includes streamlit elements that are used to build/train/predict with an esn."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

import rescomp
import rescomp.esn_new_update_code as esn

ESN_DICT = {"ESN_normal": esn.ESN_normal,
            }

ESN_HASH_FUNCS = {val: hash for val in ESN_DICT.values()}

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


@st.cache(hash_funcs=ESN_HASH_FUNCS)
def build(esn_type: str, seed: int, x_dim: int, **kwargs) -> ESN_TYPING:
    """Build the esn class.

    Args:
        esn_type: One of the esn types defined in ESN_DICT.
        seed: Set the global seed. TODO: maybe dont set global seed?
        x_dim: The x_dimension of the data to be predicted.
        **kwargs: All other build args.

    Returns:
        The build esn.
    """
    if esn_type in ESN_DICT.keys():
        esn = ESN_DICT[esn_type]()
    else:
        raise Exception("This esn_type is not accounted for")

    np.random.seed(seed)

    build_kwargs = rescomp.utilities._remove_invalid_args(esn.build, kwargs)
    esn.build(x_dim, **build_kwargs)
    return esn


def st_basic_esn_build(esn_sub_section: tuple[str, ...] | None = None,
                       key: str | None = None) -> tuple[str, dict[str, object]]:
    """Streamlit elements to specify the basic esn settings.

    Args:
        esn_sub_section: A subsection of the keys in ESN_DICT, or if None, take all of ESN_DICT.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A tuple with the first element being the esn type, the second argument being the
        build_args.
    """
    if esn_sub_section is None:
        esn_dict = ESN_DICT
    else:
        esn_dict = {esn_name: esn_class for esn_name, esn_class in ESN_DICT.items()
                    if esn_name in esn_sub_section}
        if len(esn_dict) == 0:  # TODO: proper error
            raise Exception(f"The systems in {esn_sub_section} are not accounted for.")

    esn_type = st.selectbox('esn type', esn_dict.keys())

    basic_build_args = {}
    basic_build_args["r_dim"] = st.number_input('Reservoir Dim', value=500, step=1,
                                                key=f"{key}__st_basic_esn_build__rd")
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
    log_reg_param = st.number_input('Log regulation parameter', value=-7., step=1., format="%f",
                                    key=f"{key}__st_basic_esn_build__reg")
    basic_build_args["reg_param"] = 10 ** (log_reg_param)

    return esn_type, basic_build_args


def st_network_build_args(key: str | None = None) -> dict[str, object]:
    """Streamlit elements to specify the network settings.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A dictionary containing the network build args.
    """
    network_build_args = {}
    network_build_args["n_rad"] = st.number_input('n_rad', value=0.4, step=0.1, format="%f",
                                                  key=f"{key}__st_network_build_args__nrad")
    network_build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=3.0, step=0.1,
                                                      key=f"{key}__st_network_build_args__ndeg")
    network_build_args["n_type_opt"] = st.selectbox('n_type_opt', NETWORK_TYPES,
                                                    key=f"{key}__st_network_build_args__nopt")
    return network_build_args


@st.cache(hash_funcs=ESN_HASH_FUNCS)
def train(esn_obj: ESN_TYPING, x_train: np.ndarray, t_train_sync: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """Train the esn_obj with a given x_train and t_train-sync.

    Args:
        esn_obj: The esn_obj, that has a train method.
        x_train: The np.ndarray of shape (t_train_sync_steps + t_train_steps, sys_dim)
        t_train_sync: The number of time steps used for syncing the esn before training.

    Returns:
        Tuple with the fitted output and the real output.
    """

    esn_obj.train(x_train, sync_steps=t_train_sync,
                  save_y_train=True, save_out=True)

    y_train_true = esn_obj.get_y_train()
    y_train_fit = esn_obj.get_out()

    return y_train_fit, y_train_true


@st.cache(hash_funcs=ESN_HASH_FUNCS)
def predict(esn_obj: ESN_TYPING, x_pred: np.ndarray, t_pred_sync: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Predict with the esn_obj with a given x_pred and x_pred_sync.

    Args:
        esn_obj: The esn_obj, that has a predict method.
        x_pred: The np.ndarray of shape (t_pred_sync_steps + t_pred_steps, sys_dim)
        t_pred_sync: The number of time steps used for syncing the esn before prediction.

    Returns:
        Tuple with predicted and true data, each of shape (t_pred_steps, sys_dim)
    """

    y_pred, y_pred_true = esn_obj.predict(x_pred, sync_steps=t_pred_sync)

    return y_pred, y_pred_true
