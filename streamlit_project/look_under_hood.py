import numpy as np
import streamlit as st
from app_fragments import system_simulation


if __name__ == '__main__':
    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = system_simulation.st_select_system()
        t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred = \
            system_simulation.st_select_time_steps_split_up()

        time_steps = np.sum((t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync,
                             t_pred))

    trajectory = system_simulation.simulate_trajectory(system_name, system_parameters, time_steps)
    st.write(str(trajectory.shape))
