import streamlit as st

import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.app_fragments.measures as measures


if __name__ == '__main__':
    st.header("System Simulation")
    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = syssim.st_select_system()
        time_steps = syssim.st_select_time_steps(default_time_steps=10000)

        if "dt" in system_parameters.keys():
            dt = system_parameters["dt"]
        else:
            dt = 1.0

        time_series = syssim.simulate_trajectory(system_name, system_parameters, time_steps)
        time_series = syssim.st_preprocess_simulation(time_series)

    if st.checkbox("Plot time series: "):
        syssim.st_default_simulation_plot(time_series)

    with st.expander("Measures based on data"):
        if st.checkbox("Statistical measures"):
            data_dict = {"time series": time_series}
            measures.st_statistical_measures(data_dict)

        if st.checkbox("Power spectrum"):
            data_dict = {"time series": time_series}
            measures.st_power_spectrum(data_dict, dt=dt)

    with st.expander("Measures based on the system"):
        if st.checkbox("Calculate largest lyapunov exponent: "):
            measures.st_largest_lyapunov_exponent(system_name, system_parameters)
