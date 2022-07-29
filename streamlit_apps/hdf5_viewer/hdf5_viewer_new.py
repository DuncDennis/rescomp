import streamlit as st
import pathlib
import h5py
import os

import rescomp.plotting as plot

import matplotlib.pyplot as plt
# plt.style.use('dark_background')
import seaborn as sns
sns.set_theme(style="darkgrid")
import pandas as pd
import plotly.express as px
from PIL import Image

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# @st.cache
def load_hdf5_file(file_path):
    f = h5py.File(file_path, 'r')
    return f


def plot_df_nicely(df, sweep_variable):

    # Parameters:
    x_size = 500
    y_size = int(1.4 * x_size)
    # figsize = (700, 500)
    log_x = False
    new_legend = " "
    legend_order = ["Model only",
                    "Reservoir only",
                    "Input hybrid",
                    "Output hybrid",
                    "Full hybrid",
                    ]

    def transform_func(input_str: str) -> str:
        if "full" in input_str:
            out = "Full hybrid"
        elif "input" in input_str:
            out = "Input hybrid"
        elif "output" in input_str:
            out = "Output hybrid"
        elif "normal" in input_str:
            out = "Reservoir only"
        elif "only_model" in input_str:
            out = "Model only"
        else:
            raise Exception(f"Not implemented: {input_str}")
        return out

    yaxis_title = r'$\text{Valid Time }(\lambda_\mathrm{max} t_v)$'
    font_size = 15
    legend_font_size = 11

    # Lorenz 63 r sweep:
    # xaxis_title = r'$D_r$'
    # xtick0 = 0.0
    # xdtick = 500
    # ytick0 = 0.0
    # ydtick = 5
    # xrange = [0, 1250]
    # yrange = [0, 15]

    ### if add new value (For Lorenz63 model eps=0.05):
    # uniques_sweep = df[sweep_variable].unique()
    # valid_times_add = [1.1324, ] * len(uniques_sweep)
    # valid_times_error_low_add = [0.2038, ] * len(uniques_sweep)
    # valid_times_error_high_add = [0.6341, ] * len(uniques_sweep)
    # data_dict = {"valid times": valid_times_add,
    #              "valid times error lower": valid_times_error_low_add,
    #              "valid times error upper": valid_times_error_high_add,
    #              sweep_variable: uniques_sweep,
    #              "Other Parameters": "type: only_model"}
    # df_new = pd.DataFrame(data_dict)
    # df = pd.concat([df, df_new], axis=0, ignore_index=True)
    ### end

    # Lorenz 63 eps sweep:
    # xaxis_title = r'$\epsilon$'
    # xtick0 = 0.0
    # xdtick = 0.02
    # ytick0 = 0.0
    # ydtick = 2
    # xrange = [0, 0.105]
    # yrange = [0, 12]

    # Lorenz w_in scale sweep:
    xaxis_title = r'$W_\text{in} \text{ scale: } \sigma$'
    xtick0 = 0.15
    xdtick = 0.1
    ytick0 = 0
    ydtick = 2
    xrange = None  #  [0, 0.105]
    yrange = None  #  [0, 12]

    # Chua r sweep:
    # xaxis_title = r'$D_r$'
    # xtick0 = 0.0
    # xdtick = 200
    # ytick0 = 0
    # ydtick = 2
    # xrange = None #  [0, 0.105]
    # yrange = None #  [0, 12]

    # KS eps1  r sweep:
    # xaxis_title = r'$D_r$'
    # xtick0 = 0.0
    # xdtick = 2000
    # ytick0 = 0.0
    # ydtick = 1
    # xrange = [-100, 8200]
    # yrange = [-0.2, 4.8]

    # ### if add new value (For KS model eps=1):
    # uniques_sweep = df[sweep_variable].unique()
    # valid_times_add = [0.0175, ] * len(uniques_sweep)
    # valid_times_error_low_add = [0, ] * len(uniques_sweep)
    # valid_times_error_high_add = [0, ] * len(uniques_sweep)
    # data_dict = {"valid times": valid_times_add,
    #              "valid times error lower": valid_times_error_low_add,
    #              "valid times error upper": valid_times_error_high_add,
    #              sweep_variable: uniques_sweep,
    #              "Other Parameters": "type: only_model"}
    # df_new = pd.DataFrame(data_dict)
    # df = pd.concat([df, df_new], axis=0, ignore_index=True)
    # ### end

    # # KS eps=0.1  r sweep:
    # xaxis_title = r'$D_r$'
    # xtick0 = 0.0
    # xdtick = 2000
    # ytick0 = 0.0
    # ydtick = 1
    # xrange = [-100, 8200]
    # yrange = [-0.2, 7.4]

    # ### if add new value (For KS model eps=0.1):
    # uniques_sweep = df[sweep_variable].unique()
    # valid_times_add = [0.5775, ] * len(uniques_sweep)
    # valid_times_error_low_add = [0.0306, ] * len(uniques_sweep)
    # valid_times_error_high_add = [0.1006, ] * len(uniques_sweep)
    # data_dict = {"valid times": valid_times_add,
    #              "valid times error lower": valid_times_error_low_add,
    #              "valid times error upper": valid_times_error_high_add,
    #              sweep_variable: uniques_sweep,
    #              "Other Parameters": "type: only_model"}
    # df_new = pd.DataFrame(data_dict)
    # df = pd.concat([df, df_new], axis=0, ignore_index=True)
    # ### end

    # # KS eps=0.01  r sweep:
    # xaxis_title = r'$D_r$'
    # xtick0 = 0.0
    # xdtick = 2000
    # ytick0 = 0.0
    # ydtick = 1
    # xrange = [-100, 8200]
    # yrange = [-0.2, 9.7]
    #
    # ### if add new value (For KS model eps=0.01):
    # uniques_sweep = df[sweep_variable].unique()
    # valid_times_add = [2.7300, ] * len(uniques_sweep)
    # valid_times_error_low_add = [0.2888, ] * len(uniques_sweep)
    # valid_times_error_high_add = [0.7962, ] * len(uniques_sweep)
    # data_dict = {"valid times": valid_times_add,
    #              "valid times error lower": valid_times_error_low_add,
    #              "valid times error upper": valid_times_error_high_add,
    #              sweep_variable: uniques_sweep,
    #              "Other Parameters": "type: only_model"}
    # df_new = pd.DataFrame(data_dict)
    # df = pd.concat([df, df_new], axis=0, ignore_index=True)
    # ### end

    figsize = (y_size, x_size)
    df[new_legend] = df["Other Parameters"].apply(transform_func)

    df.sort_values([new_legend, sweep_variable], inplace=True)

    # st.table(df)

    fig = px.line(df, x=sweep_variable, y="valid times", error_y="valid times error upper",
                  error_y_minus="valid times error lower",
                  color=new_legend,
                  # line_dash =new_legend,
                  # line_group=new_legend,

                  width=figsize[0],
                  height=figsize[1], log_x=log_x,
                  category_orders={new_legend: legend_order})

    if log_x:
        fig.update_layout(
            xaxis={
                'exponentformat': 'E'}
        )

    fig.update_layout(
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        xaxis=dict(
            tickmode='linear',
            tick0=xtick0,
            dtick=xdtick
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=ytick0,
            dtick=ydtick
        ),

        font=dict(
            size=font_size,
            family="Times New Roman"
        )
    )
    fig.update_yaxes(range=yrange)
    fig.update_xaxes(range=xrange)

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01, # 0.99
        xanchor="left",
        # x=0.01,
        font=dict(size=legend_font_size)
    ))

    # st.plotly_chart(fig)

    fig.write_image("test_fig.png", scale=3)
    # fig.write_image("test_fig.pdf")
    image = Image.open('test_fig.png')
    st.image(image)

    # df.reset_index(inplace=True)
    # sns.lineplot(data=df, x=sweep_variable, y="valid times", hue="Other Parameters")
    # fig = plt.gcf()
    # st.pyplot(fig)


def check_parameter_sweep(f, stripped=True):
    params_dict = {}
    for trajectory in list(f["runs"].keys()):
        for key, val in f["runs"][trajectory].attrs.items():
            if key in params_dict.keys():
                if stripped:
                    if val not in params_dict[key]:
                        params_dict[key].append(val)
                else:
                    params_dict[key].append(val)
            else:
                params_dict[key] = [val, ]
    return params_dict


def get_trajectories(multiselect_dict, f):
    for key, val in multiselect_dict.items():
        if len(val) == 0:
            st.warning(f"Choose at least one option for {key}")
            raise Exception(f"Choose at least one option for {key}")

    trajs = []
    params_to_show = []
    for trajectory in list(f["runs"].keys()):
        select_traj = True
        for key, val in f["runs"][trajectory].attrs.items():
            if val not in multiselect_dict[key]:
                select_traj = False
                break
        if select_traj:
            params_this_traj = {}
            trajs.append(trajectory)
            for key, val in multiselect_dict.items():
                if len(val)>1:
                    params_this_traj[key] = f["runs"][trajectory].attrs[key]
            params_to_show.append(params_this_traj)
    return trajs, params_to_show


repo_path = pathlib.Path(__file__).parent.resolve().parents[1]
path = pathlib.Path.joinpath(repo_path, "results")
print(path)

# hdf5_files = [x for x in os.listdir(path) if x.endswith(".hdf5")]
hdf5_files = sorted(pathlib.Path(path).iterdir(), key=os.path.getmtime)[::-1]
hdf5_files = [x.name.split(".")[0] for x in hdf5_files if x.name.endswith(".hdf5")]

with st.sidebar:
    option = st.sidebar.selectbox(
        'Select experiment', hdf5_files)

    st.markdown("""---""")

st.subheader(f"{option}")
st.markdown("""---""")
# option_best = st.sidebar.button("Get best Parameters")
# TODO: Add this functionality

if option is not None:
    f = load_hdf5_file(pathlib.Path.joinpath(path, option+".hdf5"))
    data_shape = f["runs"]["trajectory_1"][:].shape
    N_ens = data_shape[0]
    N_time_periods = data_shape[1]
    with st.sidebar:

        st.write(f"N_ens: {N_ens}")
        st.sidebar.write(f"N_time_periods: {N_time_periods}")

    params_dict = check_parameter_sweep(f)

    multiselect_dict = {}
    select_all_string = "SELECT ALL"
    with st.sidebar:
        for i, (key, val) in enumerate(params_dict.items()):
            disabled = True if len(val) == 1 else False
            if not disabled:
                container = st.container()
                all = st.checkbox("Select all", key=str(i))
                if all:
                    default = val
                else:
                    default = val[0]
                multiselect_dict[key] = container.multiselect(key, val, default=default, disabled=disabled)
            else:
                multiselect_dict[key] = st.multiselect(key, val, default=val[0], disabled=disabled)

    trajs, params_to_show = get_trajectories(multiselect_dict, f)

    with st.container():
        plot_vt = st.checkbox("plot valid times heatmap")
        if plot_vt:
            error_threshhold = st.slider("error threshhold", min_value=0.00001, max_value=3., step=0.001, value=0.5)
            fig = plot.plot_valid_times_heatmap(trajs, params_to_show, f, error_threshhold=error_threshhold)
            st.pyplot(fig)
        st.markdown("""---""")

    with st.container():
        plot_error = st.checkbox("plot error")

        if plot_error:
            st.header("Plot Error over Time:")

            left, right = st.columns(2)
            with left:
                error_bar = st.checkbox("show error bars")
            with right:
                ylog = st.checkbox('y log plot')

            # max_value = f["runs"][trajs[0]][:].shape[-2]
            # max_x = st.slider("max_x", min_value=5, max_value=max_value, step=1, value=500)

            fig = plot.plot_error_plotly(trajs, params_to_show, f, error_bar=error_bar, ylog=ylog)
            st.plotly_chart(fig)

            # fig = plot.plot_error(trajs, params_to_show, f, max_x, error_bar=error_bar, ylog=ylog)
            # st.pyplot(fig)
        st.markdown("""---""")

    with st.expander("Trajectory Measures: "):
        left, mid, right = st.columns(3)
        with left:
            plot_attractor = st.checkbox("plot attractor")
        with mid:
            plot_trajectory = st.checkbox("plot trajectory")
        with right:
            plot_corrdim = st.checkbox("plot correlation dimension")

        left_low, right_low = st.columns(2)
        with left_low:
            plot_poincare = st.checkbox("plot poincare type map")
        with right_low:
            plot_lyap = st.checkbox("plot lyapunov")
        # st.markdown("""---""")
        if plot_attractor or plot_trajectory or plot_corrdim or plot_poincare or plot_lyap:
            st.markdown("""---""")
            left, right = st.columns(2)
            with left:
                i_ens = st.number_input("i_ens", min_value=0, max_value=N_ens-1)
            with right:
                i_time_period = st.number_input("i_time_period", min_value=0, max_value=N_time_periods-1)

            if plot_attractor:
                st.markdown("""---""")
                st.header("Plot Attractor: ")
                fig = plot.plot_attractor_2(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(15, 4))
                st.pyplot(fig)

            if plot_trajectory:
                st.markdown("""---""")
                N_dims = f["runs"][trajs[0]][:].shape[-1]
                st.header("Plot Trajectory: ")
                i_dim = st.number_input("dimension", min_value=0, max_value=N_dims-1)
                fig = plot.plot_trajectories(trajs, params_to_show, f, i_ens, i_time_period, i_dim, base_fig_size=(15, 4))
                st.pyplot(fig)

            if plot_corrdim:
                st.markdown("""---""")
                st.header("Plot Correlation Dimension")
                nr_steps = st.slider("Nr steps for correlation dim", min_value=1, max_value=600, step=1, value=10)
                fig = plot.plot_correlation_dimension_hdf5(trajs, params_to_show, f, i_ens, i_time_period, figsize=(8, 4),
                                                      nr_steps=nr_steps)
                st.pyplot(fig)

            if plot_poincare:
                st.markdown("""---""")
                st.header("Plot Poincare map")
                figs = plot.plot_poincare_type_map_plotly_hdf5(trajs, params_to_show, f, i_ens, i_time_period,
                                                               figsize=(20, 30), alpha=0.5, s=3)
                for params, plotly_fig in figs.items():
                    st.write(params)
                    st.plotly_chart(plotly_fig)

            if plot_lyap:
                st.markdown("""---""")
                pass
        # st.markdown("""---""")

    with st.expander("Histograms"):
        plot_corrdim_hist = st.checkbox("plot correlation dimension HISTOGRAM")
        if plot_corrdim_hist:
            st.header("Plot Correlation Dimension Histogram:")
            nr_steps = st.slider("Nr steps for correlation dim (hist)", min_value=1, max_value=600, step=1, value=10)
            bins = st.slider("histogram bins", min_value=1, max_value=100, step=1, value=10)
            fig = plot.plot_correlation_dimension_hist(trajs, params_to_show, f, nr_steps=nr_steps,
                                                       base_figsize=(15, 4), bins=bins)
            st.pyplot(fig)

        plot_valid_times_hist = st.checkbox("plot valid times HISTOGRAM")
        if plot_valid_times_hist:
            st.header("Plot Valid Times Histogram:")

            ensemble_along = st.selectbox("Ensemble along ...", ["Combined", "Network Realizations", "Time Periods"])
            i_ens_disabled = True
            i_time_period_disabled = True
            if ensemble_along == "Network Realizations":
                i_time_period_disabled = False
            elif ensemble_along == "Time Periods":
                i_ens_disabled = False
            left, right = st.columns(2)
            with left:
                i_ens = st.number_input("i_ens", min_value=0, max_value=N_ens-1, disabled=i_ens_disabled, key="i_ens2")
            with right:
                i_time_period = st.number_input("i_time_period", min_value=0, max_value=N_time_periods-1,
                                                disabled=i_time_period_disabled, key="i_time_period2")
            if ensemble_along == "Network Realizations":
                i_ens = None
            elif ensemble_along == "Time Periods":
                i_time_period = None
            elif ensemble_along == "Combined":
                i_ens, i_time_period = None, None

            error_threshhold = st.number_input("error_threshhold", min_value=0.01, max_value=5., value=0.4, step=0.1)
            fig = plot.plot_valid_times_histogram(trajs, params_to_show, f, i_ens=i_ens, i_time_period=i_time_period,
                                                  error_threshhold=error_threshhold, figsize=(800, 500))
            st.plotly_chart(fig)

    with st.expander("Valid Times vs. Parameter"):
        plot_valid_times_vs_param = st.checkbox("plot Valid Times vs. Params")
        if plot_valid_times_vs_param:
            st.header("Plot Valid Times vs. Parameter:")

            multiselect_dict_only_multiples = {key: val for key, val in multiselect_dict.items() if len(val)>1}

            sweep_variable = st.selectbox("sweep variable", multiselect_dict_only_multiples.keys())

            #####
            ensemble_along = st.selectbox("Ensemble along ...", ["Combined", "Network Realizations", "Time Periods"], key="2")
            i_ens_disabled = True
            i_time_period_disabled = True
            if ensemble_along == "Network Realizations":
                i_time_period_disabled = False
            elif ensemble_along == "Time Periods":
                i_ens_disabled = False
            left, right = st.columns(2)
            with left:
                i_ens = st.number_input("i_ens", min_value=0, max_value=N_ens - 1, disabled=i_ens_disabled, key="i_ens3")
            with right:
                i_time_period = st.number_input("i_time_period", min_value=0, max_value=N_time_periods - 1,
                                                disabled=i_time_period_disabled, key="i_time_period3")
            if ensemble_along == "Network Realizations":
                i_ens = None
            elif ensemble_along == "Time Periods":
                i_time_period = None
            elif ensemble_along == "Combined":
                i_ens, i_time_period = None, None

            # Experimental:
            # lyapunov_exponents = rescomp.simulations.standard_lyapunov_exponents
            # if st.checkbox("In lyapunov times", disabled=disabled):
            #     in_lyapunov_times = {"dt": dt, "LE": lyapunov_exponents[system_option]}
            # else:
            #     in_lyapunov_times = None

            if st.checkbox("In lyapunov times"):
                left, right = st.columns(2)
                with left:
                    dt = st.number_input("dt", value=0.1)
                    lyapunov_exponent = st.number_input("LLE", value=0.9059, format="%f")
                in_lyapunov_times = {"dt": dt, "LE": lyapunov_exponent}
            else:
                in_lyapunov_times = None

            log_x = st.checkbox("log_x")
            average_type = st.selectbox("average_type", ["mean", "median"])

            #####

            # if len(data_shape) == 2:  # Data is already valid times
            #     fig = plot.plot_vt(trajs, params_to_show, f, sweep_variable,
            #                        i_ens=i_ens, i_time_period=i_time_period,
            #                        figsize=(800, 500), log_x=log_x,
            #                        average_type=average_type,
            #                        in_lyapunov_times=in_lyapunov_times)
            # elif len(data_shape) == 3:  # Data is error.
            #     pass  # TBD data is error.
            # else:  # Data is trajectory:
            #     error_threshhold = st.number_input("error_threshhold", min_value=0.01,
            #                                        max_value=5.,
            #                                        value=0.4, key="error2", step=0.1)
            #     fig = plot.plot_valid_times_sweep(trajs, params_to_show, f, sweep_variable,
            #                                       i_ens=i_ens, i_time_period=i_time_period,
            #                                       error_threshhold=error_threshhold,
            #                                       figsize=(800, 500), log_x=log_x,
            #                                       average_type=average_type,
            #                                       in_lyapunov_times=in_lyapunov_times)

            if len(data_shape) == 2:  # Data is already valid times
                disabled = True
            else:
                disabled = False

            error_threshhold = st.number_input("error_threshhold", min_value=0.01,
                                               max_value=5.,
                                               value=0.4, key="error2", step=0.1, disabled=disabled)

            df = plot.hdf5_to_pandas_vt(trajs, params_to_show, f, sweep_variable, i_ens=i_ens,
                                        i_time_period=i_time_period,
                                        error_threshhold=error_threshhold,
                                        average_type=average_type,
                                        in_lyapunov_times=in_lyapunov_times)

            fig = plot.plot_pandas(df, sweep_variable=sweep_variable, figsize=(800, 500),
                                   log_x=log_x)
            st.plotly_chart(fig)

            if st.checkbox("Table: ", value=False):
                st.table(df)

            if st.checkbox("Plot dataframe nicely"):
                plot_df_nicely(df, sweep_variable)
            # csv = convert_df(df)
            #
            # st.download_button("Download DataFrame as CSV", data=csv, mime="text/csv")

    with st.expander("Attractor Likeness vs. Parameter"):
        plot_attr_likeness_vs_param = st.checkbox("plot Attractor Likeness vs. Params")
        if plot_attr_likeness_vs_param:
            st.header("Plot Attractor Likeness vs. Params:")

            multiselect_dict_only_multiples = {key: val for key, val in multiselect_dict.items() if len(val)>1}

            sweep_variable = st.selectbox("sweep variable", multiselect_dict_only_multiples.keys(), key="al1")

            #####
            ensemble_along = st.selectbox("Ensemble along ...", ["Combined", "Network Realizations", "Time Periods"], key="al2")
            i_ens_disabled = True
            i_time_period_disabled = True
            if ensemble_along == "Network Realizations":
                i_time_period_disabled = False
            elif ensemble_along == "Time Periods":
                i_ens_disabled = False
            left, right = st.columns(2)
            with left:
                i_ens = st.number_input("i_ens", min_value=0, max_value=N_ens - 1, disabled=i_ens_disabled, key="al3")
            with right:
                i_time_period = st.number_input("i_time_period", min_value=0, max_value=N_time_periods - 1,
                                                disabled=i_time_period_disabled, key="al4")
            if ensemble_along == "Network Realizations":
                i_ens = None
            elif ensemble_along == "Time Periods":
                i_time_period = None
            elif ensemble_along == "Combined":
                i_ens, i_time_period = None, None

            fig = plot.plot_attr_likeness_sweep(trajs, params_to_show, f, sweep_variable, i_ens=i_ens, i_time_period=i_time_period,
                                                  bins=100, figsize=(800, 500))
            st.plotly_chart(fig)

            # fig = plot.plot_valid_times_sweep_error_first(trajs, params_to_show, f, sweep_variable, i_ens=i_ens, i_time_period=i_time_period,
            #                                       error_threshhold=error_threshhold, figsize=(800, 500))
            # st.plotly_chart(fig)
