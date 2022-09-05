"""A streamlit app to demonstrate PCA is conjunction with Echo State Networks - the Intro page."""

import streamlit as st

st.set_page_config("Intro", page_icon="👋")

st.header("ESN and PCA Demo")

st.markdown(
    """
    This app aims to demonstrate and explain how **Principal Component Analysis (PCA)** 
    can be used in conjunction with an **Echo State Network (ESN)**. 
    
    **Summary of the pages:** 
    - 📐: Explain PCA, and demonstrate it on an example. 
    - 📈: Explain multiple linear regression in conjunction with pca, demonstrate it on an example.
    - 💫: Apply PCA and on the ESN and see its effects. 
    """
)
