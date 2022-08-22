import numpy as np

import streamlit as st

from streamlit_project.app_fragments import streamlit_utilities as utils
from streamlit_project.app_fragments import esn_build_train_predict as esnbuild



def st_remove_r_gen_and_drive(esn_obj,):
    # TBD:
    options = list(np.arange())
    st.multiselect("R_gen dims", options)


def correlate_input_and_r_gen(inp: np.ndarray, r_gen: np.ndarray, time_delay: int
                              ) -> np.ndarray:
    """Correlate the reservoir input with the driven r_gen states. Add a timedelay. """
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
