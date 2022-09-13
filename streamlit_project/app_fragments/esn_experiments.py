import numpy as np

import streamlit as st

from streamlit_project.app_fragments import streamlit_utilities as utils
from streamlit_project.app_fragments import esn_build_train_predict as esnbuild


def correlate_input_and_r_gen(inp: np.ndarray, r_gen: np.ndarray, time_delay: int
                              ) -> np.ndarray:
    """Correlate the reservoir input with the driven r_gen states. Add a timedelay.
    # TODO: proper docstring """
    if time_delay == 0:
        r_gen_slice = r_gen
        inp = inp
    else:
        if time_delay > 0:
            r_gen_slice = r_gen[time_delay:, :]
            inp = inp[:-time_delay, :]
        elif time_delay < 0:
            r_gen_slice = r_gen[:time_delay, :]
            inp = inp[-time_delay:, :]
        else:
            raise ValueError

    r_gen_dim = r_gen_slice.shape[1]
    inp_dim = inp.shape[1]
    correlation = np.zeros((r_gen_dim, inp_dim))
    for i_r in range(r_gen_dim):
        for i_inp in range(inp_dim):
            correlation[i_r, i_inp] = \
                np.corrcoef(r_gen_slice[:, i_r], inp[:, i_inp])[0, 1]

    return correlation


@st.cache(hash_funcs=esnbuild.ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def drive_reservoir(esn_obj: esnbuild.ESN_TYPING, x_drive: np.ndarray
                    ) -> tuple[np.ndarray, esnbuild.ESN_TYPING]:
    """Drive reservoir and output r_gen.
    # TODO: experimental
    Args:
        esn_obj: The esn_obj, that has a predict method.
        x_drive: The np.ndarray to drive the resrvoir with.

    Returns:
        Tuple with r_gen driven and the esn_obj.
    """

    esn_obj.reset_r()
    esn_obj.drive(x_drive, save_r_gen=True)

    r_gen_driven = esn_obj.get_r_gen()

    return r_gen_driven, esn_obj


@st.cache(hash_funcs=esnbuild.ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def train_on_subsets(esn_obj: esnbuild.ESN_TYPING,
                     x_train: np.ndarray,
                     t_train_sub: int,
                     t_train_sync_sub: int,
                     seperate_factor: float = 0.5
                     ):
    """Train the esn_obj on multiple subsets of x_train.
    # TODO: not finished what to output.
    """

    if seperate_factor < 0.1:
        raise ValueError("seperate factor has to be bigger than 0.1")

    total_time_steps = x_train.shape[0]
    x_train_subs = []
    starting_step = 0
    while True:
        end_step = starting_step + (t_train_sub + t_train_sync_sub)
        if end_step >= total_time_steps:
            break
        x_train_sub = x_train[starting_step: end_step, :]
        x_train_subs.append(x_train_sub)
        starting_step += int((t_train_sub + t_train_sync_sub) * seperate_factor)

    r_gen_train_list = []
    w_out_list = []
    for i_x, x_train_sub in enumerate(x_train_subs):
        esn_obj.train(x_train_sub,
                      sync_steps=t_train_sync_sub,
                      save_y_train=True,
                      save_out=True,
                      save_res_inp=True,
                      save_r_internal=True,
                      save_r=True,
                      save_r_gen=True
                      )
        r_gen_train_list.append(esn_obj.get_r_gen())
        w_out_list.append(esn_obj.get_w_out())
    return r_gen_train_list, w_out_list
