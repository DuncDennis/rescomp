"""Utlity streamlit fragments."""
from __future__ import annotations

import streamlit as st


def line() -> None:
    """Draw a seperator line."""
    st.markdown("""---""")


def dimension_selection(dimension: int, key: str | None = None) -> int:
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
