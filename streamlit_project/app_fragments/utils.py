"""Utlity streamlit fragments."""
from __future__ import annotations

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


if __name__ == '__main__':
    pass
