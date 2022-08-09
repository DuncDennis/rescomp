"""Utlity streamlit fragments."""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st


def st_line() -> None:
    """Draw a seperator line."""
    st.markdown("""---""")


def st_selectbox_with_all(name: str, options: list[str], key: str | None = None) -> list[str]:
    """A streamlit element for a multiselect with a "select all" checkbox.

    Args:
        name: The name of the multiselect.
        options: The options in the multiselect.
        key: A optional key if it's used multiple times.

    Returns:
        The selection.
    """
    container = st.container()
    all = st.checkbox("Select all", key=f"{key}__select_all")
    if all:
        default = options
    else:
        default = options[0]

    selection = container.multiselect(name, options, default=default, key=f"{key}__multi_select")

    return selection


def st_dimension_selection(dimension: int, key: str | None = None) -> int:
    """A number input to select the dimension.

    Args:
        dimension: The dimension of the system.
        key: A possible unique key.

    Returns:
        The selected dimension.
    """

    return st.number_input("Dimension", value=0, max_value=dimension-1,
                           min_value=0, key=f"{key}__dim_selection")


def st_dimension_selection_multiple(dimension: int, key: str | None = None) -> list[int]:
    """Streamlit element to select multiple dimensions.

    Args:
        dimension: The maximal dimension that can be selected.

    Returns:
        A list of integers representing the selected dimensions.
    """

    dim_select_opts = [f"{i}" for i in range(dimension)]
    dim_selection = st_selectbox_with_all("Dimensions", dim_select_opts,
                                          key=f"{key}__dim_select_mult")
    dim_selection = [int(x) for x in dim_selection]
    return dim_selection


def st_main_checkboxes(key: str | None = None) -> tuple[bool, bool, bool, bool]:
    """Streamlit element to create 4 esn checkboxes: Simulate data, Build, Train and Predict.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        The states of the four checkboxes: simulate_bool, build_bool, train_bool, predict_bool.
    """

    basic_key = f"{key}__st_main_checkboxes"

    def on_change_sim():
        if not st.session_state[f"{basic_key}__sim"]:
            other_keys = [f"{basic_key}__{x}" for x in ("build", "train", "pred")]
            for k in other_keys:
                st.session_state[k] = False

    def on_change_build():
        if not st.session_state[f"{basic_key}__build"]:
            other_keys = [f"{basic_key}__{x}" for x in ("train", "pred")]
            for k in other_keys:
                st.session_state[k] = False

    def on_change_train():
        if not st.session_state[f"{basic_key}__train"]:
            other_keys = [f"{basic_key}__{x}" for x in ("pred", )]
            for k in other_keys:
                st.session_state[k] = False

    simulate_bool = st.checkbox("🌀Simulate data", key=f"{basic_key}__sim",
                                on_change=on_change_sim)

    disabled = False if simulate_bool else True
    build_bool = st.checkbox("🛠️ Build", disabled=disabled, key=f"{basic_key}__build",
                             on_change=on_change_build)

    disabled = False if build_bool else True
    train_bool = st.checkbox("🦾 Train", disabled=disabled, key=f"{basic_key}__train",
                             on_change=on_change_train)

    disabled = False if train_bool else True
    predict_bool = st.checkbox("🔮 Predict", disabled=disabled, key=f"{basic_key}__pred")

    return simulate_bool, build_bool, train_bool, predict_bool


@st.experimental_memo
def get_random_int() -> int:
    """Get a random integer between 1 and 1000000.
    TODO: maybe handle with generators in future.
    Is used to get a new seed.

    Returns:
        The random integer.
    """
    return np.random.randint(1, 1000000)


def st_seed(key: str | None = None) -> int:
    """Streamlit element to specify the random seed.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        The seed.
    """
    new_seed = st.button("new random seed", key=f"{key}__new_seed")
    if new_seed:
        get_random_int.clear()
        seed = get_random_int()
        st.session_state[f"{key}__st_seed__seed"] = seed

    seed = st.number_input("Seed", max_value=1000000, key=f"{key}__st_seed__seed")
    return seed


def st_add_to_state(prefix: str, name: str, value: Any) -> None:
    """Add a variable to the session state.

    The name will be saved as f"{prefix}__{name}"

    Args:
        prefix: The prefix of the session state variable.
        name: The name of the session state variable.
        value: The value of the variable.

    """
    full_name = f"{prefix}__{name}"
    st.session_state[full_name] = value


def st_get_session_state(prefix: str, name: str) -> Any:
    """Get a variable of session state by defining the prefix and name.

    Args:
        prefix: The prefix of the session state variable.
        name: The name of the session state variable.

    Returns:
        The value of the variable.
    """
    full_name = f"{prefix}__{name}"
    if full_name in st.session_state:
        return st.session_state[full_name]
    else:
        return None


def st_reset_all_check_boxes(key: str | None = None) -> None:
    if st.button("Untick all", key=f"{key}__st_reset_all_check_boxes"):
        true_checkboxes = {key: val for key, val in st.session_state.items() if
                           type(val) == bool and val is True}
        for k in true_checkboxes.keys():
            if k == f"{key}__st_reset_all_check_boxes":
                continue
            else:
                st.session_state[k] = False


if __name__ == '__main__':
    pass
