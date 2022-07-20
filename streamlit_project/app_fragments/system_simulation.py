import streamlit as st
import numpy as np

import rescomp.simulations_new as sims


name_dict = {
    "Lorenz96": sims.Lorenz96,
    "Roessler": sims.Roessler,
    "ComplexButterly": sims.ComplexButterly,
    "Chen": sims.Chen,
    "ChuaCircuit": sims.ChuaCircuit,
    "Thomas": sims.Thomas,
    "WindmiAttractor": sims.WindmiAttractor,
    "Rucklidge": sims.Rucklidge,
    "SimplestQuadraticChaotic": sims.SimplestQuadraticChaotic,
    "SimplestCubicChaotic": sims.SimplestCubicChaotic,
    "SimplestPiecewiseLinearChaotic": sims.SimplestPiecewiseLinearChaotic,
    "UedaOscillator": sims.UedaOscillator,
    "Henon": sims.Henon,
    "Logistic": sims.Logistic
}


# SYSTEMS = ["Lorenz63", "Roessler", "Logistic"]


def select_system():
    out = {}
    out["system"] = st.selectbox('System', SYSTEMS)
    out["dt"] = st.number_input('dt', value=0.05, step=0.01, format="%f")


    return out

# with st.sidebar:
#     # System to predict:
#     st.header("System: ")
#
#     system_option = st.sidebar.selectbox(
#         'System to Predict', systems_to_predict)
#     dt = st.number_input('dt', value=0.05, step=0.01, format="%f")
#     with st.expander("Time Steps: "):
#         t_train_disc = int(st.number_input('t_train_disc', value=1000, step=1))
#         t_train_sync = int(st.number_input('t_train_sync', value=300, step=1))
#         t_train = int(st.number_input('t_train', value=5000, step=1))
#         t_pred_disc = int(st.number_input('t_pred_disc', value=1000, step=1))
#         t_pred_sync = int(st.number_input('t_pred_sync', value=300, step=1))
#         t_pred = int(st.number_input('t_pred', value=5000, step=1))
#         all_time_steps = int(t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred)
#         time_boundaries = (0, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred)
#         st.write(f"Total Timessteps: {all_time_steps}, Maximal Time: {np.round(dt * all_time_steps, 1)}")


if __name__ == '__main__':
    with st.sidebar:
        out = select_system()
