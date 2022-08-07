"""Utlity streamlit fragments."""
from __future__ import annotations

import streamlit as st


def st_line() -> None:
    """Draw a seperator line."""
    st.markdown("""---""")


def st_selectbox_with_all(name: str, options: list[str], key: None = None) -> list[str]:
    """A streamlit element for a multiselect with a "select all" checkbox.

    Args:
        name: The name of the multiselect.
        options: The options in the multiselect.
        key: A optional key if it's used multiple times.

    Returns:
        The selection.
    """
    container = st.container()
    all = st.checkbox("Select all", key=key)
    if all:
        default = options
    else:
        default = options[0]

    return container.multiselect(name, options, default=default, key=key)


def st_dimension_selection(dimension: int, key: str | None = None) -> int:
    """A number input to select the dimension.

    Args:
        dimension: The dimension of the system.
        key: A possible unique key.

    Returns:
        The selected dimension.
    """
    return st.number_input("Dimension", value=0, max_value=dimension-1,
                           min_value=0, key=key)


if __name__ == '__main__':
    pass
