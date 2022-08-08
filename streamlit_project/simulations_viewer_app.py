import streamlit as st

import streamlit_project.app_fragments.system_simulation as syssim
import streamlit_project.app_fragments.measures as measures
import streamlit_project.app_fragments.utils as utils
import streamlit_project.app_fragments.plotting as plot


if __name__ == '__main__':
    st.set_page_config("Dynamical Systems Viewer", page_icon="🦋")

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
        data_dict = {"time series": time_series}
        # data_dict = {"time series": time_series,
        #              "time series2": time_series*0.9}

    with st.expander("Plotting"):
        utils.st_line()
        if st.checkbox("Plot"):
            plot.st_default_simulation_plot(time_series)
        utils.st_line()
        if st.checkbox("Plot 1D"):
            plot.st_plot_dim_selection(data_dict)
        utils.st_line()
        if st.checkbox("Plot 3D delay embedding"):
            plot.st_one_dim_time_delay(data_dict)

    with st.expander("Measures based on data"):
        utils.st_line()
        if st.checkbox("Consecutive extrema"):
            measures.st_extrema_map(data_dict)
        utils.st_line()
        if st.checkbox("Statistical measures"):
            measures.st_statistical_measures(data_dict)
        utils.st_line()
        if st.checkbox("Histogram"):
            measures.st_histograms(data_dict)
        utils.st_line()
        if st.checkbox("Power spectrum"):
            measures.st_power_spectrum(data_dict, dt=dt)
        utils.st_line()
        if st.checkbox("Lyapunov from data"):
            measures.st_largest_lyapunov_from_data(data_dict, dt=dt)
    with st.expander("Measures based on the system"):
        utils.st_line()
        if st.checkbox("Calculate largest lyapunov exponent"):
            measures.st_largest_lyapunov_exponent(system_name, system_parameters)
