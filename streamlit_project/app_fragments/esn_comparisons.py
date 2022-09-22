"""A python file for functions that are used to compare different ESNs"""
from __future__ import annotations

from typing import Any

import copy
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_project.app_fragments import streamlit_utilities as utils
from streamlit_project.app_fragments import esn_build_train_predict as esnbuild
from streamlit_project.app_fragments import esn_plotting as esnplot


def compare_esn_parameters(different_esn_parameters: dict[str, Any],
                           mode: str = "difference"
                           ) -> None | pd.DataFrame:
    """Compare a dict of parameters dicts.

    Args:
        different_esn_parameters: A dictionary containing for each key (the esn name) a dictionary
                                  for its parameters.
        mode: Either "difference" or "all". If "difference" the output df contains only parameters
              that are not the same for all esns. If "all" all build args are outputted.

    Returns:
        A pandas DataFrame containing the index "Parameters", and for each esn_key a column.
    """
    if len(different_esn_parameters) == 0:
        return None
    dep = different_esn_parameters
    for i, (key, val) in enumerate(dep.items()):
        pd_dict = {"Parameters": list(val.keys()), key: [str(x) for x in list(val.values())]}
        df = pd.DataFrame.from_dict(pd_dict).set_index("Parameters")
        if i == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], axis=1, join="outer")

    if mode == "difference":
        x = df_all.index[~df_all.eq(df_all.iloc[:, 0], axis=0).all(1)]
        df_all = df_all.loc[x]
    elif mode == "all":
        pass
    else:
        raise ValueError("mode must be either \"difference\" or \"all\".")

    return df_all


def st_comparison_build_train_predict(esn_type: str,
                                      build_args: dict[str, Any],
                                      esn_obj: esnbuild.ESN_TYPING,
                                      y_train_fit: np.ndarray,
                                      y_train_true: np.ndarray,
                                      res_train_dict: dict[str, np.ndarray],
                                      y_pred: np.ndarray,
                                      y_pred_true: np.ndarray,
                                      res_pred_dict: dict[str, np.ndarray],
                                      w_out: np.ndarray,
                                      system_name: str,
                                      system_parameters: dict[str, Any],
                                      scale_shift_vector: None | tuple[np.ndarray, np.ndarray],
                                      seed: int,
                                      x_dim: int,
                                      x_train: np.ndarray,
                                      t_train_sync: int,
                                      x_pred: np.ndarray,
                                      t_pred_sync: int,
                                      compare_esn_parms_container: st.container | None = None,
                                      key: str | None = None
                                      ) -> tuple[dict[str, dict[str, Any]],
                                                 dict[str, dict[str, Any]],
                                                 dict[str, esnbuild.ESN_TYPING]]:
    """Streamlit element to build additional ESNs and train and predict with them.

    Args:
        esn_type: The name of the base esn.
        build_args: The build args of the base esn.
        esn_obj: The base ens_object.
        y_train_fit: The fitted y_train of the base esn.
        y_train_true: The true y_train of the base esn.
        res_train_dict: The reservoir states dict of the base esn during training.
        y_pred: The predicted trajectory of the base esn.
        y_pred_true: The true trajectory of the base esn.
        res_pred_dict: The reservoir states dict of the base esn during prediction.
        w_out: w_out of the base esn.
        system_name: The system name of the data.
        system_parameters: The system parameters of the data.
        scale_shift_vector: The scale_shift_vector of the preprocessing of the data.
        seed: The random seed used for build and train.
        x_dim: The dimension of the data.
        x_train: The training data.
        t_train_sync: The synchronization time steps for training.
        x_pred: The data to predict.
        t_pred_sync: The synchronization time steps for prediction.
        compare_esn_parms_container: An optional contained to display the differences in esn
                                     parameters.
        key: An optional key if this function is used multiple times.

    Returns: A tuple of three dictionaries:
        1. A dictionary with an entry for each esn, where one of them is the base esn with the key
        "base". Each value for each esn, is a dictionary with the keys "train", "predict" and
        "w_out". The first two contain a 3-tuple: (pred/fit, true, res_states). The "w_out" key
        contains the w_out matrix.
        2. A dictionary with a pca_parameter dictionary for each key.
        3. A dictionary with the esn_obj for each key.
    """
    nr_of_comparisons = int(st.number_input("Nr. of comparisons",
                                            value=0,
                                            min_value=0,
                                            key=f"{key}__st_comparison_build_train_predict__nr"))

    esn_parameters = build_args.copy()
    esn_parameters["esn_type"] = esn_type
    different_esn_parameters = {}
    different_esn_parameters["base"] = esn_parameters

    different_esn_outputs = {}
    different_esn_outputs["base"] = {"train": (y_train_fit, y_train_true, res_train_dict),
                                     "predict": (y_pred, y_pred_true, res_pred_dict),
                                     "w_out": w_out}

    different_esn_objects = {}
    different_esn_objects["base"] = copy.deepcopy(esn_obj)

    for i_comp in range(nr_of_comparisons):
        default_name = str(i_comp + 1)
        with st.expander(f"Parameters of comparison ESN: {default_name}"):
            name = st.text_input("Name",
                                 value=default_name,
                                 key=f"{key}__st_comparison_build_train_predict__{default_name}")
            if name in different_esn_outputs.keys():
                raise ValueError("Name already taken.")
            utils.st_line()

            # BUILD parameters
            st.markdown("**ESN type:**")
            esn_type = esnbuild.st_select_esn_type(key=default_name)
            st.markdown("**Basic parameters:**")
            build_args = esnbuild.st_basic_esn_build(key=default_name)
            st.markdown("**Network parameters:**")
            build_args = build_args | esnbuild.st_network_build_args(key=default_name)

            if esn_type == "ESN_r_process":
                st.markdown("**ESN_r_process settings:**")
                build_args = build_args | esnbuild.st_esn_r_process_args(build_args["r_dim"],
                                                                         key=default_name)

            if esn_type == "ESN_pca":
                st.markdown("**ESN_pca settings:**")
                build_args = build_args | esnbuild.st_pca_build_args(build_args["r_dim"],
                                                                     key=default_name)
            if esn_type == "ESN_hybrid":
                st.markdown("**ESN_hybrid settings:**")
                build_args = build_args | esnbuild.st_hybrid_build_args(
                    system_name=system_name,
                    system_parameters=system_parameters,
                    key=default_name)
                build_args["scale_shift_vector_input"] = scale_shift_vector
                build_args["scale_shift_vector_output"] = scale_shift_vector
            utils.st_line()

            # BUILD
            esn_obj = esnbuild.build(esn_type, seed=seed, x_dim=x_dim, build_args=build_args)
            esn_obj = copy.deepcopy(esn_obj)

            # TRAIN
            y_train_fit, y_train_true, res_train_dict, esn_obj = esnbuild.train_return_res(esn_obj,
                                                                                           x_train,
                                                                                           t_train_sync,
                                                                                           )
            esn_obj = copy.deepcopy(esn_obj)

            # PREDICT
            y_pred, y_pred_true, res_pred_dict, esn_obj = esnbuild.predict_return_res(esn_obj,
                                                                                      x_pred,
                                                                                      t_pred_sync)
            different_esn_outputs[name] = {
                "train": (y_train_fit, y_train_true, res_train_dict),
                "predict": (y_pred, y_pred_true, res_pred_dict),
                "w_out": esn_obj.get_w_out()}

            esn_parameters = build_args.copy()
            esn_parameters["esn_type"] = esn_type
            different_esn_parameters[name] = esn_parameters

            different_esn_objects[name] = copy.deepcopy(esn_obj)

    # Summary of different build args:
    if compare_esn_parms_container is not None:
        with compare_esn_parms_container:
            with st.expander("Differences in ESN parameters"):
                df_parameters_comparison = compare_esn_parameters(different_esn_parameters,
                                                                  mode="difference")
                st.table(df_parameters_comparison)

    return different_esn_outputs, different_esn_parameters, different_esn_objects


def transform_to_r_gen_w_out(different_esn_outputs) -> dict[str, dict[str, Any]]:
    """Transform the output from st_comparison_build_train_predict to get r_gen and w_out.

    Get the input to be used in st_pca_transformed_quantities_comp.

    Args:
        different_esn_outputs: Output from st_comparison_build_train_predict

    Returns:
        A dictionary which has the same keys as different_esn_outputs, but each value is
        now a dict with the keys "r_gen_train, "r_gen_pred", and "w_out".
    """
    r_gen_w_out_dict_comp = {}
    for k, v in different_esn_outputs.items():
        r_gen_w_out_dict_comp[k] = {"r_gen_train": v["train"][2]["r_gen"],
                                    "r_gen_pred": v["predict"][2]["r_gen"],
                                    "w_out": v["w_out"]}
    return r_gen_w_out_dict_comp


def st_pca_transformed_quantites_comp(r_gen_w_out_dict_comp: dict[str, dict[str, Any]],
                                      key: str | None = None
                                      ) -> dict[str, dict[str, Any]]:
    """Streamlit element to pca transform r_gen and w_out from the r_gen_w_out_dict_comp.

    Args:
        r_gen_w_out_dict_comp: The output from transform_to_r_gen_w_out.
        key: A unique key, if this function is called multiple times.

    Returns:
        The same structure as the input, but the r_gen's and the w_out is pca transformed.
    """

    st.markdown(r"Choose whether you want to perform an additional "
                r"*PCA-transformation* on $R_\text{gen}$ and $W_\text{out}$ before "
                r"the analysis.")
    choice = st.radio("PCA before analysis?",
                      ["no", "yes"],
                      key=f"{key}__st_pca_transformed_quantites")

    if choice == "no":
        return r_gen_w_out_dict_comp

    elif choice == "yes":
        with st.expander("More info..."):
            st.markdown(
                r"""
                **Perform a Principal Component Analysis on the** $R_\text{gen}$ **states:**
                - Fit the PCA on the $R_\text{gen, train}$ states. 
                - Use the fitted PCA to transform the $R_\text{gen, pred}$ states. 
                - Obtain $R_\text{gen, train}^\text{pca}$ and $R_\text{gen, pred}^\text{pca}$.
                - Transform $W_\text{out}$ to $W_\text{out}^\text{pca}$ with the PC-Matrix $P$.
                In the following, $r_\text{gen}$ refers to $r_\text{gen, pca}$ and
                $W_\text{out}$ refers to $W_\text{out}^\text{pca}$.
                """)
        r_gen_w_out_dict_pca_comp = {}
        for k, v in r_gen_w_out_dict_comp.items():
            out = esnplot.get_pca_transformed_quantities(r_gen_train=v["r_gen_train"],
                                                         r_gen_pred=v["r_gen_pred"],
                                                         w_out=v["w_out"])
            r_gen_w_out_dict_pca_comp[k + " (pca)"] = {"r_gen_train": out[0],
                                                       "r_gen_pred": out[1],
                                                       "w_out": out[2]}
        return r_gen_w_out_dict_pca_comp

    else:
        raise ValueError("This choice is not accounted for. ")
