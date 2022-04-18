import streamlit as st
import numpy as np
import rescomp
import rescomp.plotting as plot
import rescomp.statistical_tests as stat_test
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# Functions:
def build_rc(ndim, w_in_scale):
    esn = rescomp.ESNWrapper()
    w_out = np.zeros((3, ndim))
    esn.create_architecture(ndim, 3, w_out=w_out, seed=None, w_out_fit_flag="simple",
                            custom_act_fct=custom_act_fct, w_in_scale=w_in_scale,
                            w_in_ordered=w_in_ordered, w_in_sparse=True)
    return esn


systems_to_predict = ["Lorenz", "Roessler"]

with st.sidebar:

    # System to predict:
    system_option = st.sidebar.selectbox(
        'System to Predict', systems_to_predict)
    dt = st.number_input('dt', value=0.01, step=0.01)
    normalize = st.checkbox('normalize')
    st.markdown("""---""")

    # Build RC architecture:
    ndim = st.number_input('Reservoir Dim', value=200, step=1)
    w_in_scale = st.number_input('w_in_scale', value=1.0, step=0.1)

    # Other parameters: # TODO: to be implemented
    custom_act_fct = None
    w_in_ordered = True

    st.markdown("""---""")
    log_reg_param = st.number_input('regulation parameter "1e-x"', value=5, step=1)
    reg_param = 10^(-log_reg_param)
    with st.expander("Time Steps"):
        t_train_disc = st.number_input('t_train_disc', value=1000, step=1)
        t_train_sync = st.number_input('t_train_sync', value=200, step=1)
        t_train = st.number_input('t_train', value=1000, step=1)
        t_pred_disc = st.number_input('t_pred_disc', value=1000, step=1)
        t_pred_sync = st.number_input('t_pred_sync', value=100, step=1)
        t_pred = st.number_input('t_pred', value=1000, step=1)
        all_time_steps = t_train_disc + t_train_sync + t_train + \
                         t_pred_disc + t_pred_sync + t_pred
        st.write(f"Total Timessteps: {all_time_steps}, Maximal Time: {np.round(dt*all_time_steps, 1)}")
with st.container():
    # build
    build_rc_bool = st.checkbox("Build RC architecture")
    if build_rc_bool:
        st.header("Build RC architecture")

        build_args = (ndim, w_in_scale)

        esn = build_rc(*build_args)

        if st.button("rebuild"):
            esn = build_rc(*build_args)

        st.subheader("W_in quantities:")
        # with st.expander("W_in quantities:"):
        fig = plot.plot_architecture(esn, figsize=(10, 4))
        st.pyplot(fig)

        st.subheader("W quantities:")
        st.write("TBD")

        st.markdown("""---""")
        train_bool = st.checkbox("Train RC")
        if train_bool:

            if system_option == "Lorenz":
                starting_point = np.array([0, 1, 0])
                time_series = rescomp.simulations.simulate_trajectory("lorenz", dt, all_time_steps, starting_point)
            elif system_option == "Roessler":
                starting_point = np.array([0, 1, 0])
                time_series = rescomp.simulations.simulate_trajectory("roessler_sprott", dt, all_time_steps,
                                                                      starting_point)
            if normalize:
                time_series = rescomp.utilities.normalize_timeseries(time_series)

            x_train, x_pred_list = stat_test.data_simulation(time_series, t_train_disc, t_train_sync, t_train,
                                                             t_pred_disc, t_pred_sync, t_pred, nr_of_time_intervals=1,
                                                             sim_data_return=False)

            esn.train(x_train, sync_steps=t_train_sync, reg_param=reg_param, w_in_no_update=True, save_r=True,
                      save_x_train_pred=True)

            x_train_true = x_train[t_train_sync:-1]

            r_train = esn._r_train_gen
            x_train_pred = esn.x_train_pred

            st.subheader("Training: ")
            st.write(str(x_train_pred.shape))
            for i in range(3):
                # st.line_chart(data=np.stack((x_train_pred[:, i], x_train_true[:, i])).T)
                st.line_chart(data={"train fit": x_train_pred[:, i], "train true": x_train_true[:, i]})
            # plot_trajectory = st.checkbox("plot trajectory")

            st.markdown("""---""")
            test_bool = st.checkbox("Test RC")
            if test_bool:
                pass
