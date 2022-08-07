from __future__ import annotations

import streamlit as st
import numpy as np


@st.experimental_memo
def get_random_int():
    print("Get new seed")
    return np.random.randint(1, 1000000)


def seed_interface():
    if st.checkbox("custom seed"):
        seed = st.number_input("custom seed", max_value=1000000)
    else:
        if st.button("new seed"):
            get_random_int.clear()
        seed = get_random_int()  # lets see what to do with that

    st.write(f"Current seed: {seed}")
    return seed


def line():
    st.markdown("""---""")


if __name__ == '__main__':
    pass
