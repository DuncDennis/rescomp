import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# import pandas as pd
import math
import rescomp.measures as measures
import rescomp.utilities as utilities



def line_plotly_extension(error_y_mode=None, **kwargs): # Not used yet
    """Extension of `plotly.express.line` to use error bands.
    From: https://stackoverflow.com/questions/69587547/continuous-error-band-with-plotly-express-in-python
    """
    ERROR_MODES = {'bar','band','bars','bands',None}
    if error_y_mode not in ERROR_MODES:
        raise ValueError(f"'error_y_mode' must be one of {ERROR_MODES}, received {repr(error_y_mode)}.")
    if error_y_mode in {'bar','bars',None}:
        fig = px.line(**kwargs)
    elif error_y_mode in {'band','bands'}:
        if 'error_y' not in kwargs:
            raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
        figure_with_error_bars = px.line(**kwargs)
        fig = px.line(**{arg: val for arg,val in kwargs.items() if arg != 'error_y'})
        for data in figure_with_error_bars.data:
            x = list(data['x'])
            y_upper = list(data['y'] + data['error_y']['array'])
            y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
            color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))},.3)".replace('((','(').replace('),',',').replace(' ','')
            fig.add_trace(
                go.Scatter(
                    x = x+x[::-1],
                    y = y_upper+y_lower[::-1],
                    fill = 'toself',
                    fillcolor = color,
                    line = dict(
                        color = 'rgba(255,255,255,0)'
                    ),
                    hoverinfo = "skip",
                    showlegend = False,
                    legendgroup = data['legendgroup'],
                    xaxis = data['xaxis'],
                    yaxis = data['yaxis'],
                )
            )
        # Reorder data as said here: https://stackoverflow.com/a/66854398/8849755
        reordered_data = []
        for i in range(int(len(fig.data)/2)):
            reordered_data.append(fig.data[i+int(len(fig.data)/2)])
            reordered_data.append(fig.data[i])
        fig.data = tuple(reordered_data)
    return fig


#### Some functions to plot a trajectory results numpy array:
def plot_attractor(out, N_ens=None, nr_of_time_intervals=None):
    print(out.shape)
    if N_ens is None:
        N_ens = out.shape[0]
    if nr_of_time_intervals is None:
        nr_of_time_intervals = out.shape[1]
    y_pred_all = out[:, :, 0, :, :]
    y_test_all = out[:, :, 1, :, :]

    # N_ens = 2
    # nr_of_time_intervals = 2

    fig, axs = plt.subplots(N_ens, nr_of_time_intervals, figsize=(5*nr_of_time_intervals, 5*N_ens))

    for i_ens in range(N_ens):
        for i_interval in range(nr_of_time_intervals):
            ax = axs[i_ens, i_interval]
            y_pred = y_pred_all[i_ens, i_interval, :, :]
            y_test = y_test_all[i_ens, i_interval, :, :]
            ax.plot(y_test[:, 0], y_test[:, 2], linewidth=1, label="true")
            ax.plot(y_pred[:, 0], y_pred[:, 2], linewidth=1, label="pred")
            ax.set_title(f"ENS: {i_ens}, Timeinterval: {i_interval}")
            ax.legend()
    return fig


def plot_difference(trajs, params_to_show, f):
    fig = plt.figure()
    ax = plt.gca()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        diff = data[:, :, 0, :, :] - data[:, :, 1, :, :]
        mean_diff = np.mean(diff, axis=(0, 1))
        plt.plot(mean_diff[:, 0], label=f"{params_to_show[i_traj]}")
    plt.legend()
    return fig


def plot_error(trajs, params_to_show, f, max_x, error_bar=False, ylog=False):
    fig = plt.figure()
    ax = plt.gca()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        error = get_error(data)
        mean_error = np.mean(error, axis=(0, 1))

        ax.plot(mean_error[:max_x], label=f"{params_to_show[i_traj]}")
        if error_bar:
            color = ax.get_lines()[-1].get_color()
            std_error = np.std(error, axis=(0, 1))
            ax.errorbar(np.arange(max_x), mean_error[:max_x], yerr=std_error[:max_x], color=color, #, label=f"{params_to_show[i_traj]}"
                        alpha=0.4)
        if ylog:
            # ax.set_xscale("log")
            ax.set_yscale("log")
    plt.legend()
    return fig


def plot_error_plotly(trajs, params_to_show, f, error_bar=False, ylog=False):
    fig = make_subplots(rows=1, cols=1, subplot_titles=(("error")))

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        error = get_error(data)
        mean_error = np.mean(error, axis=(0, 1))
        x = np.arange(mean_error.shape[0])
        name = f"{params_to_show[i_traj]}"
        if error_bar:
            std_error = np.std(error, axis=(0, 1))
            error_y = dict(
            type='data', # value of error bar given in data coordinates
            array=std_error,
            visible=True)
        else:
            error_y = None

        fig.add_trace(
            go.Scatter(x=x, y=mean_error, error_y=error_y, mode="lines", name=name)
        )
        if ylog:
            fig.update_yaxes(type="log")

    return fig


def plot_attractor_2(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(5, 10)):
    fig, axs = plt.subplots(len(trajs), 1, figsize=(base_fig_size[0], base_fig_size[1]*len(trajs)))

    for i_traj, traj in enumerate(trajs):
        try:
            ax = axs[i_traj]
        except:
            ax = axs
        data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        y_pred = data[0, :, :]
        y_test = data[1, :, :]
        ax.plot(y_test[:, 0], y_test[:, 2], linewidth=1, label="true")
        ax.plot(y_pred[:, 0], y_pred[:, 2], linewidth=1, label="pred")
        ax.set_title(f"{params_to_show[i_traj]}")
        ax.legend()
    return fig


def plot_trajectories(trajs, params_to_show, f, i_ens, i_time_period, i_dim, base_fig_size=(5, 10)):
    fig, axs = plt.subplots(len(trajs), 1, figsize=(base_fig_size[0], base_fig_size[1]*len(trajs)))

    for i_traj, traj in enumerate(trajs):
        try:
            ax = axs[i_traj]
        except:
            ax = axs
        data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        y_pred = data[0, :, :]
        y_test = data[1, :, :]
        ax.plot(y_test[:, i_dim], label="true")
        ax.plot(y_pred[:, i_dim], label="pred")
        ax.set_title(f"{params_to_show[i_traj]}")
        ax.legend()
    return fig


# Utility:
def get_error(data):
    """
    data like:  data = f["runs"][traj][:]
    """
    n_ens = data.shape[0]
    n_interval = data.shape[1]
    n_pred_steps = data.shape[3]

    error = np.zeros((n_ens, n_interval, n_pred_steps))
    for i_ens in range(n_ens):
        for i_interval in range(n_interval):
            y_pred = data[i_ens, i_interval, 0, :, :]
            y_test = data[i_ens, i_interval, 1, :, :]

            error[i_ens, i_interval, :] = measures.error_over_time(y_pred, y_test, distance_measure="L2",
                                                                    normalization="root_of_avg_of_spacedist_squared")
    return error


def get_valid_times(data, error_threshhold=1.0):
    """
    data like:  data = f["runs"][traj][:]
    """
    error = get_error(data)
    n_ens, n_interval = data.shape[0], data.shape[1]
    valid_times = np.zeros((n_ens, n_interval))
    for i_ens in range(n_ens):
        for i_interval in range(n_interval):
            valid_times[i_ens, i_interval] = measures.valid_time_index(error[i_ens, i_interval, :],
                                                                       error_threshhold)
    return valid_times


def plot_valid_times_heatmap(trajs, params_to_show, f, error_threshhold, base_fig_size=(5, 10)):
    fig, axs = plt.subplots(1, len(trajs), figsize=(base_fig_size[0]*len(trajs), base_fig_size[1]))
    # v_min = None
    # v_max = None
    for i_traj, traj in enumerate(trajs):
        try:
            ax = axs[i_traj]
        except:
            ax = axs
        data = f["runs"][traj][:]
        n_ens, n_interval = data.shape[0], data.shape[1]
        error = get_error(data)

        valid_times = np.zeros((n_ens, n_interval))
        for i_ens in range(n_ens):
            for i_interval in range(n_interval):
                valid_times[i_ens, i_interval] = measures.valid_time_index(error[i_ens, i_interval, :],
                                                                           error_threshhold)

        # if v_min is None:
        #     v_min = np.min(valid_times)
        # else:
        #     v_min_new = np.min(valid_times)
        #     if v_min < v_min_new:
        #         v_min = v_min_new

        im = ax.imshow(valid_times)
        ax.set_xlabel("i_time_period")
        ax.set_ylabel("i_ens")
        ax.set_title(f"{params_to_show[i_traj]}")

    # fig.colorbar(im)
    return fig


def plot_correlation_dimension_hdf5(trajs, params_to_show, f, i_ens, i_time_period, figsize=(5, 10), nr_steps=10):
    fig = plt.figure(figsize=figsize)

    corr_dim_true = None

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        y_pred = data[0, :, :]
        corr_dim_pred = measures.dimension(y_pred, return_neighbours=True, nr_steps=nr_steps)

        if corr_dim_true is None:
            y_test = data[1, :, :]
            corr_dim_true = measures.dimension(y_test, return_neighbours=True, nr_steps=nr_steps)

        plt.loglog(corr_dim_pred[1][0], corr_dim_pred[1][1],
                   label=f"pred: {np.round(corr_dim_pred[0], 3)}, {params_to_show[i_traj]}")

    plt.loglog(corr_dim_true[1][0], corr_dim_true[1][1],
               label=f"true: {np.round(corr_dim_true[0], 3)}", linestyle="--", c="r")
    plt.legend()
    return fig


def plot_correlation_dimension_hist(trajs, params_to_show, f, base_figsize=(5, 10), nr_steps=10, bins=10):
    fig, axs = plt.subplots(len(trajs), 1, figsize=(base_figsize[0], base_figsize[1]*len(trajs)))

    # true:
    data = f["runs"][trajs[0]][:]
    N_ens = data.shape[0]
    N_time_periods = data.shape[1]
    y_test = data[0, 0, 1, :, :]
    cor_dim_array_true = measures.dimension(y_test, return_neighbours=False, nr_steps=nr_steps)

    for i_traj, traj in enumerate(trajs):
        try:
            ax = axs[i_traj]
        except:
            ax = axs
        data = f["runs"][traj][:]

        # N_ens = data.shape[0]
        # N_time_periods = data.shape[1]
        cor_dim_array = np.zeros((N_ens, N_time_periods))

        for i_ens in range(N_ens):
            for i_t in range(N_time_periods):
                y_pred = data[i_ens, i_t, 0, :, :]
                cor_dim_array[i_ens, i_t] = measures.dimension(y_pred, return_neighbours=False, nr_steps=nr_steps)

        ax.hist(cor_dim_array.flatten(), bins=bins)
        ax.set_title(f"{params_to_show[i_traj]}")
        ax.axvline(cor_dim_array_true, linestyle="--", c="r")

    return fig


# For "look under hood":
def plot_architecture(w_in, figsize=(10, 5)):
    """
    Plot w_in distribution and plot
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # w_in = esn._w_in
    w_in_flat = w_in.flatten()
    n_dim, x_dim = w_in.shape

    max_val, min_val = np.max(w_in_flat), np.min(w_in_flat)
    max_abs_val = np.max([np.abs(max_val), np.abs(min_val)])
    for i_x in range(x_dim):
        y_left = i_x
        for i_n in range(n_dim):
            y_right = i_n/(n_dim/x_dim)

            val = w_in[i_n, i_x]

            # c = "r" if val > 0 else "b"
            # axs[0].plot([0, 1], [y_left, y_right], c=f"{c}", linewidth=np.abs(val))  # , marker="."

            val_norm = (val - min_val)/(max_val - min_val)
            c = (val_norm, 1-val_norm, 0, np.abs(val)/(max_abs_val))
            axs[0].plot([0, 1], [y_left, y_right], c=c, linewidth=np.abs(val)/(max_abs_val))  # , marker="."

    axs[0].axis('off')
    axs[0].set_title("W_in connection")

    w_in_non_zero = w_in_flat[w_in_flat != 0]
    w_in_non_zero_mean = w_in_non_zero.mean()
    axs[1].hist(w_in_non_zero, bins="auto")
    axs[1].axvline(w_in_non_zero_mean, c="r", linestyle="--")
    axs[1].set_title(f"W_in value histogram, non-zero mean: {np.round(w_in_non_zero_mean, 4)}")

    return fig


def plot_w_out(w_out, figsize=(10, 5)):
    """
    Plot w_out distribution and plot
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # w_in = esn._w_in
    w_out_flat = w_out.flatten()
    n_dim, x_dim = w_out.shape

    max_val, min_val = np.max(w_out_flat), np.min(w_out_flat)
    max_abs_val = np.max([np.abs(max_val), np.abs(min_val)])
    for i_x in range(x_dim):
        y_left = i_x
        for i_n in range(n_dim):
            y_right = i_n/(n_dim/x_dim)

            val = w_out[i_n, i_x]

            # c = "r" if val > 0 else "b"
            # axs[0].plot([0, 1], [y_left, y_right], c=f"{c}", linewidth=np.abs(val))  # , marker="."

            val_norm = (val - min_val)/(max_val - min_val)
            c = (val_norm, 1-val_norm, 0, np.abs(val)/(max_abs_val))
            axs[0].plot([0, 1], [y_left, y_right], c=c, linewidth=np.abs(val)/(max_abs_val))  # , marker="."

    axs[0].axis('off')
    axs[0].set_title("W_out connection")

    w_out_non_zero = w_out_flat[w_out_flat != 0]
    w_out_non_zero_mean = w_out_non_zero.mean()
    axs[1].hist(w_out_non_zero, bins="auto")
    axs[1].axvline(w_out_non_zero_mean, c="r", linestyle="--")
    axs[1].set_title(f"W_out value histogram, non-zero mean: {np.round(w_out_non_zero_mean, 4)}")

    return fig


## plotly:
def plot_3d_time_series(time_series, line=True):
    x = time_series[:, 0]
    y = time_series[:, 1]
    z = time_series[:, 2]
    if line:
        fig = px.line_3d(x=x, y=y, z=z)
    else:
        fig = px.scatter_3d(x=x, y=y, z=z)
    return fig


def plot_3d_time_series_multiple(time_series_dict, mode="line", size=None):
    to_plot_dict = {"x": [], "y": [], "z": [], "label": []}
    for label, time_series in time_series_dict.items():
        to_plot_dict["x"].extend(time_series[:, 0])
        to_plot_dict["y"].extend(time_series[:, 1])
        to_plot_dict["z"].extend(time_series[:, 2])
        to_plot_dict["label"].extend([label, ] * time_series.shape[0])

    if mode == "line":
        fig = px.line_3d(to_plot_dict, x="x", y="y", z="z", color="label")
    elif mode == "scatter":
        fig = px.scatter_3d(to_plot_dict, x="x", y="y", z="z", color="label")
        if size is not None:
            fig.update_traces(marker={'size': size})
    return fig


def plot_1d_time_series(time_series, i_dim, boundaries):
    names = ["train disc", "train sync", "train", "pred disc", "pred sync", "pred"]
    to_plot = time_series[:, i_dim]
    x_list = np.arange(time_series.shape[0])
    if boundaries is not None:
        right = 0
        fig = make_subplots()
        for i in range(6):
            left = right
            right = boundaries[i+1] + right

            fig.add_trace(go.Scatter(
                x=x_list[left: right],
                y=to_plot[left: right],
                name=names[i] #line_shape='hv'
            )
            )
    return fig


def plot_multiple_1d_time_series(time_series_data, title=""):
    fig = px.line(time_series_data, title=title)
    return fig


def show_reservoir_states(res_states):
    # TODO rename show_image
    fig = px.imshow(res_states.T, aspect="auto")
    return fig


def plot_log_divergence(log_div_list, dt=1.0, fit=True, t_min=None, t_max=None, figsize=(9, 4), ax=None, label=None):
    time_steps = log_div_list.size
    t_list = np.arange(time_steps) * dt

    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    round_digs = 5

    ax.plot(t_list, log_div_list, label=label)
    c = ax.lines[-1].get_color()
    if fit:
        if t_min is None:
            t_min = 0
        if t_max is None:
            t_max = t_list[-1]

        x_fit, y_fit, coefs = utilities._linear_fit(log_div_list, dt=dt, t_min=t_min, t_max=t_max)

        ax.plot(x_fit, y_fit,
                label=f"Sloap = {np.round(coefs[0], round_digs)}, Intersect = {np.round(coefs[1], round_digs)}",
                linestyle="--", c=c)

    ax.grid()
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(r"avg. log distance")
    # ax.set_title(title)
    if return_fig:
        return fig


def plot_node_value_histogram(states, ax=None, title="", figsize=(8, 3)):
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        return_fig = False

    data = states.flatten()
    ax.hist(data, bins="auto")
    ax.set_title(title)
    if return_fig:
        return fig


def plot_node_value_histogram_multiple(states_data_dict, act_fct=None, figsize=(15, 7)):
    nr_of_hists = len(states_data_dict.keys())

    ncols = 2
    nrows = math.ceil(nr_of_hists/ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    for i, (title, states) in enumerate(states_data_dict.items()):
        i_col = i % ncols
        i_row = int(i/ncols)

        ax = axs[i_row, i_col]
        plot_node_value_histogram(states, ax=ax, title=title)
        if title == "act_fct_inp":
            if act_fct is not None:
                ax_twin = ax.twinx()
                x_lim_low, x_lim_high = ax.get_xlim()
                x_range = np.arange(x_lim_low, x_lim_high, 0.05)
                ax_twin.plot(x_range, act_fct(x_range), c="r")
    return fig


def plot_error_single(y_pred, y_true, title=""):
    error_over_time = measures.error_over_time(y_pred, y_true, distance_measure="L2",
                                               normalization="root_of_avg_of_spacedist_squared")
    fig = px.line(error_over_time, title=title)
    return fig


def plot_image_and_timeseries(inp_res, upd_res, res, time_series, figsize=(13, 5)):
    fig, axs = plt.subplots(nrows=3+3, ncols=1, figsize=figsize)

    for i in range(3):
        ax = axs[i]
        ax.plot(time_series[:, i])
        ax.grid(True)
    ax = axs[3]
    ax.imshow(inp_res.T, aspect="auto")
    ax.set_title("inp_res")
    ax.grid(True)

    ax = axs[4]
    ax.imshow(upd_res.T, aspect="auto")
    ax.set_title("upd_res")
    ax.grid(True)

    ax = axs[5]
    ax.imshow(res.T, aspect="auto")
    ax.set_title("res")
    ax.grid(True)

    return fig

def plot_correlation_dimension(y_pred, y_true, nr_steps=20, r_min=1.5, r_max=5., figsize=(13, 5)):
    fig = plt.figure(figsize=figsize)

    sloap_true, N_r = measures.dimension(y_true, return_neighbours=True, nr_steps=nr_steps, r_min=r_min,
                                         r_max=r_max)
    plt.loglog(N_r[0], N_r[1], label=f"True: {sloap_true}")

    sloap_pred, N_r = measures.dimension(y_pred, return_neighbours=True, nr_steps=nr_steps, r_min=r_min,
                                         r_max=r_max)
    plt.loglog(N_r[0], N_r[1], label=f"Pred: {sloap_pred}")

    plt.legend()
    plt.xlabel("radius")
    plt.ylabel("Nr of Points")
    return fig


def plot_poincare_type_map(y_pred, y_true, dim=None, mode="maxima", value_or_time="value", figsize=(13, 5), s=1.0, alpha=1.0):
    if value_or_time == "value":
        val_bool = True
    elif value_or_time == "time":
        val_bool = False
    else:
        raise Exception(f"value_or_time not recognized: {value_or_time}")

    x_dim = y_pred.shape[1]
    if dim is None:
        dims = list(np.arange(x_dim))
    else:
        dims = (dim, )

    dims_to_show = len(dims)

    fig, axs = plt.subplots(nrows=dims_to_show, ncols=1, figsize=figsize)
    if dims_to_show == 1:
        axs = (axs, )

    for i_d, d in enumerate(dims):
        ax = axs[i_d]

        if val_bool:
            x, y = measures.poincare_map(y_true, dimension=d, mode=mode)
        else:
            x, y = measures.poincare_map_for_time(y_true, dimension=d, mode=mode)
        ax.scatter(x, y, label="True", s=s, alpha=alpha)
        if val_bool:
            x, y = measures.poincare_map(y_pred, dimension=d, mode=mode)
        else:
            x, y = measures.poincare_map_for_time(y_pred, dimension=d, mode=mode)
        ax.scatter(x, y, label="Pred", s=s, alpha=alpha)

        ax.set_title(f"Dimension: {d}")
        ax.legend()

    return fig


def plot_poincare_type_map_plotly(y_pred, y_true, dim=None, mode="maxima", value_or_time="value", figsize=(13, 5),
                                  s=1.0, alpha=1.0):

    if value_or_time == "value":
        val_bool = True
    elif value_or_time == "time":
        val_bool = False

    x_dim = y_pred.shape[1]
    if dim is None:
        dims = list(np.arange(x_dim))
    else:
        dims = (dim, )

    dims_to_show = len(dims)

    fig = make_subplots(rows=dims_to_show, cols=1, subplot_titles=([f"dim {x}" for x in range(dims_to_show)]))

    # fig, axs = plt.subplots(nrows=dims_to_show, ncols=1, figsize=figsize)
    # if dims_to_show == 1:
    #     axs = (axs, )

    for i_d, d in enumerate(dims):
        if val_bool:
            x, y = measures.poincare_map(y_true, dimension=d, mode=mode)
        else:
            x, y = measures.poincare_map_for_time(y_true, dimension=d, mode=mode)
        fig.add_trace(
            go.Scatter(x=x, y=y, opacity=alpha, name="True", mode='markers', marker=dict(color="lightgreen")),
            row=i_d+1, col=1
        )
        fig.update_traces(marker={'size': s})
        # ax.scatter(x, y, label="True", s=s, alpha=alpha)

        if val_bool:
            x, y = measures.poincare_map(y_pred, dimension=d, mode=mode)
        else:
            x, y = measures.poincare_map_for_time(y_pred, dimension=d, mode=mode)
        fig.add_trace(
            go.Scatter(x=x, y=y, opacity=alpha, name="Pred", mode='markers', marker=dict(color="red")),
            row=i_d+1, col=1
        )
        # ax.scatter(x, y, label="Pred", s=s, alpha=alpha)

        # ax.set_title(f"Dimension: {d}")
        # ax.legend()
    fig.update_traces(marker={'size': s})
    fig.update_layout(height=figsize[1]*30, width=figsize[0]*30)
    return fig


def plot_model_likeness(y_pred, iterator, steps=10, figsize=(15, 4)):
    error = measures.model_likeness(y_pred, iterator, steps=steps)

    error_sum = np.sum(error)

    fig = plt.figure(figsize=figsize)


    plt.plot(error)
    plt.xlabel("steps")
    plt.ylabel("Avg L2 Distance: True iterator vs Y_pred")
    plt.title(f"error sum: {error_sum}")
    plt.grid()
    return fig


def plot_val_vs_next_val(data, figsize=(13, 6), alpha=1, size=1):
    fig = plt.figure(figsize=figsize)
    for key, y in data.items():
        plt.scatter(y[:-1], y[1:], label=key, alpha=alpha, s=size)

    plt.xlabel("x(t)")
    plt.ylabel("x(t+1)")
    plt.legend()
    return fig


def plot_lyapunov_spectrum(time_series, dt=1.0, freq_cut=True, pnts_to_try=50, steps=100, figsize=(13, 5), ax=None,
                           label=None):
    return_fig = False
    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    avg_log_dist, t_list = measures.lyapunov_rosenstein(time_series, dt=dt, freq_cut=freq_cut, pnts_to_try=pnts_to_try,
                                                        steps=steps)
    plot_log_divergence(avg_log_dist, dt=dt, fit=True, t_min=None, t_max=None, ax=ax, label=label)

    if return_fig:
        return fig


def plot_lyapunov_spectrum_multiple(data, dt=1.0, freq_cut=True, pnts_to_try=50, steps=100, figsize=(13, 5)):
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    for label, time_series in data.items():
        # print("timeseiresshape: ", time_series.shape)
        plot_lyapunov_spectrum(time_series, dt=dt, freq_cut=freq_cut, pnts_to_try=pnts_to_try,
                                                        steps=steps, ax=ax, label=label)

    return fig


def plot_poincare_type_map_plotly_hdf5(trajs, params_to_show, f, i_ens, i_time_period, **kwargs):
    figs = {}
    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        params = params_to_show[i_traj]
        y_pred = data[0, :, :]
        y_true = data[1, :, :]

        fig = plot_poincare_type_map_plotly(y_pred, y_true, **kwargs)
        figs[str(params)] = fig
    return figs


def plot_valid_times_histogram(trajs, params_to_show, f, i_ens=None, i_time_period=None, figsize=(150, 150),
                               error_threshhold=0.4):
    df = pd.DataFrame()
    means = []
    for i_traj, traj in enumerate(trajs):
        # data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        data = f["runs"][traj][:]
        valid_times = get_valid_times(data, error_threshhold=error_threshhold)
        params_str = str(params_to_show[i_traj])
        if i_ens is None and i_time_period is None:
            valid_times = valid_times.flatten()
        elif i_ens is None and i_time_period is not None:
            valid_times = valid_times[:, i_time_period]
        elif i_ens is not None and i_time_period is None:
            valid_times = valid_times[i_ens, :]

        df_to_add = pd.DataFrame()
        df_to_add["valid times"] = valid_times
        df_to_add["parameters"] = params_str

        df = pd.concat([df, df_to_add])
        means.append(np.mean(valid_times))

    fig = px.histogram(df, x="valid times", color="parameters", width=figsize[0], height=figsize[1],
                       opacity=0.5)
    for i_m, mean in enumerate(means):
        color = fig.data[i_m].marker.color
        fig.add_vline(x=mean, line_dash="dash", line_color=color)
    return fig


def plot_valid_times_sweep(trajs, params_to_show, f, sweep_variable, i_ens=None, i_time_period=None, figsize=(150, 150),
                               error_threshhold=0.4, log_x=False, average_type="mean"):
    """
    For hdf5 viewer: plot valid times vs a sweep variable
    """
    df = pd.DataFrame()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        params = params_to_show[i_traj]

        valid_times = get_valid_times(data, error_threshhold=error_threshhold)

        if i_ens is None and i_time_period is None:
            valid_times = valid_times.flatten()
        elif i_ens is None and i_time_period is not None:
            valid_times = valid_times[:, i_time_period]
        elif i_ens is not None and i_time_period is None:
            valid_times = valid_times[i_ens, :]

        if average_type == "mean":
            valid_times_avg = np.mean(valid_times)
            valid_times_error_lower = np.std(valid_times)
            valid_times_error_upper = valid_times_error_lower

        elif average_type == "median":
            valid_times_avg = np.median(valid_times)
            valid_times_error_lower = valid_times_avg - np.quantile(valid_times, q=0.25)
            valid_times_error_upper = np.quantile(valid_times, q=0.75) - valid_times_avg

        df_to_add = pd.DataFrame()
        df_to_add["valid times"] = [valid_times_avg]
        df_to_add["valid times error lower"] = [valid_times_error_lower]
        df_to_add["valid times error upper"] = [valid_times_error_upper]
        df_to_add[sweep_variable] = [params[sweep_variable]]

        params_str = ", ".join([f"{key}: {val}" for key, val in params.items() if key != sweep_variable])
        df_to_add["Other Parameters"] = [params_str]

        df = pd.concat([df, df_to_add])

    print(df)
    df.sort_values(["Other Parameters", sweep_variable], inplace=True)
    fig = px.line(df, x=sweep_variable, y="valid times", error_y="valid times error upper",
                  error_y_minus="valid times error lower", color="Other Parameters", width=figsize[0],
                  height=figsize[1], log_x=log_x)

    if log_x:
        fig.update_layout(
            xaxis={
                'exponentformat': 'E'}
        )
    return fig


def plot_valid_times_sweep_error_first(trajs, params_to_show, f, sweep_variable, i_ens=None, i_time_period=None, figsize=(150, 150),
                               error_threshhold=0.4):
    """
    Same as plot_valid_times_sweep but the mean error is calculated first, and then the "mean" valid time
    """
    df = pd.DataFrame()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        params = params_to_show[i_traj]
        error = get_error(data)

        if i_ens is None and i_time_period is None:
            error_mean = np.mean(error, axis=(0, 1))
        elif i_ens is None and i_time_period is not None:
            error_mean = np.mean(error[:, i_time_period, :], axis=0)
        elif i_ens is not None and i_time_period is None:
            error_mean = np.mean(error[i_ens, :, :], axis=0)

        valid_times_mean = measures.valid_time_index(error_mean, error_threshhold)

        # valid_times_mean = np.mean(valid_times)
        # valid_times_std = np.std(valid_times)
        # print(valid_times_mean)
        df_to_add = pd.DataFrame()
        df_to_add["valid times"] = [valid_times_mean]
        # df_to_add["valid times std"] = [valid_times_std]
        df_to_add[sweep_variable] = [params[sweep_variable]]
        print(df_to_add)
        params_str = ", ".join([f"{key}: {val}" for key, val in params.items() if key != sweep_variable])
        # del params[sweep_variable]
        # params_str = str(params)
        df_to_add["Other Parameters"] = [params_str]

        df = pd.concat([df, df_to_add])

    print(df)
    df.sort_values(["Other Parameters", sweep_variable], inplace=True)
    fig = px.line(df, x=sweep_variable, y="valid times", color="Other Parameters", width=figsize[0],
                  height=figsize[1],)
    return fig


def show_hist(data, bins=100, figsize=(15, 8)):
    # assume data is a dict of the kind: {"r_pred_3dim": np.array(1000, 3), "r_..." ..)
    fig, axs = plt.subplots(3, 1, figsize=figsize)
    for i_ax, ax in enumerate(axs):
        ax.set_title(f"axes: {i_ax}")
        for label, traj in data.items():
            ax.hist(traj[:, i_ax], alpha=0.5, label=label, bins=bins)
            ax.legend()

    return fig


def show_res_state_scatter(res_states, figsize=(15, 8), sort=True, s=0.1, alpha=0.5):
    # assume data is a dict of the kind: {"r_pred_3dim": np.array(1000, 3), "r_..." ..)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if sort:
        pass
    n_points, r_dim = res_states.shape
    y = np.repeat(np.arange(0, r_dim), n_points)  # array like [0,0,0, 1,1,1, 2,2,2, ..]
    x = res_states.flatten()

    ax.scatter(x, y, s=s, alpha=alpha)

    return fig


def get_attractor_likeness(data, bins=100):
    n_ens = data.shape[0]
    n_interval = data.shape[1]
    n_pred_steps = data.shape[3]

    attr_likeness = np.zeros((n_ens, n_interval, n_pred_steps))
    for i_ens in range(n_ens):
        for i_interval in range(n_interval):
            y_pred = data[i_ens, i_interval, 0, :, :]
            y_test = data[i_ens, i_interval, 1, :, :]

            attr_likeness[i_ens, i_interval, :] = measures.attractor_likeness(y_test, y_pred, bins=bins)

    return attr_likeness


def plot_attr_likeness_sweep(trajs, params_to_show, f, sweep_variable, i_ens=None, i_time_period=None, figsize=(150, 150),
                               bins=100):
    """
    For hdf5 viewer: plot valid times vs a sweep variable
    """
    df = pd.DataFrame()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        params = params_to_show[i_traj]

        attr_likeness = get_attractor_likeness(data, bins=bins)

        if i_ens is None and i_time_period is None:
            attr_likeness = attr_likeness.flatten()
        elif i_ens is None and i_time_period is not None:
            attr_likeness = attr_likeness[:, i_time_period]
        elif i_ens is not None and i_time_period is None:
            attr_likeness = attr_likeness[i_ens, :]

        attr_likeness_mean = np.mean(attr_likeness)
        attr_likeness_std = np.std(attr_likeness)
        # print(attr_likeness_mean)
        df_to_add = pd.DataFrame()
        df_to_add["attr likeness mean"] = [attr_likeness_mean]
        df_to_add["attr likeness std"] = [attr_likeness_std]
        df_to_add[sweep_variable] = [params[sweep_variable]]
        print(df_to_add)
        # del params[sweep_variable]
        # params_str = str(params)
        params_str = ", ".join([f"{key}: {val}" for key, val in params.items() if key != sweep_variable])
        df_to_add["Other Parameters"] = [params_str]

        df = pd.concat([df, df_to_add])

    print(df)
    df.sort_values(["Other Parameters", sweep_variable], inplace=True)
    fig = px.line(df, x=sweep_variable, y="attr likeness mean", error_y="attr likeness std", color="Other Parameters", width=figsize[0],
                  height=figsize[1],)
    return fig


def plot_wout_magnitudes(w_out, figsize=(150, 150)):
    # not needed anymore
    x_dim, r_gen_dim = w_out.shape

    data_dict = {"index": [], "magnitude": [], "out_channel": []}

    for i_x in range(x_dim):
        for i_n in range(r_gen_dim):
            w_mag = np.abs(w_out[i_x, i_n])

            data_dict["index"].append(i_n)
            data_dict["magnitude"].append(w_mag)
            data_dict["out_channel"].append(i_x)

    df = pd.DataFrame.from_dict(data_dict)
    fig = px.bar(df, x="index", y="magnitude", color="out_channel", title="W_out magnitudes", width=figsize[0],
                 height=figsize[1])
    return fig


def plot_state_std(data, figsize=(150, 150), title=""):
    # not needed anymore
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

    for name, vals in data.items():
        std = np.std(vals, axis=0)
        fig.add_trace(
            go.Bar(y=std, name=f"std of {name}")
        )

    fig.update_layout(height=figsize[1], width=figsize[0])
    return fig


def plot_w_out_and_r_gen_std_quantites(r_gen_data, w_out, figsize=(650, 500), log_y=False):
    """
    Ultimate w_out and r_gen value plotting func. This includes also the functionality of "plot_state_std" and "plot_wout_magnitudes"
    """
    figs = []

    x_dim, r_gen_dim = w_out.shape
    data_dict = {"r_gen_index": [], "w_out": [], "out_channel": [],  "std_r_gen": [], "type": []} #"mean_r_gen": [],

    for t, r_gen in r_gen_data.items():
        # mean_r_gen = np.mean(r_gen, axis=0)
        std_r_gen = np.std(r_gen, axis=0)
        for i_x in range(x_dim):
            for i_n in range(r_gen_dim):
                w = w_out[i_x, i_n]
                data_dict["r_gen_index"].append(i_n)
                data_dict["w_out"].append(w)
                data_dict["out_channel"].append(i_x)

                # data_dict["mean_r_gen"].append(mean_r_gen[i_n])
                data_dict["std_r_gen"].append(std_r_gen[i_n])

                data_dict["type"].append(t)

    df = pd.DataFrame.from_dict(data_dict)

    df["w_out_magnitude"] = np.abs(df["w_out"])
    df["r_gen_std_times_w_out"] = np.abs(df["std_r_gen"] * df["w_out"])

    r_gen_data_types = list(r_gen_data.keys())

    # w_out figure:
    df_w_out = df[df["type"] == r_gen_data_types[0]]
    fig = px.bar(df_w_out, x="r_gen_index", y="w_out_magnitude", color="out_channel",
                 title="W_out magnitudes per out-channel", width=figsize[0],
                 height=figsize[1], log_y=log_y)
    if log_y:
        fig.update_layout(
            yaxis={
                'exponentformat': 'E'}
        )
    figs.append(fig)

    # r_gen_std figure:
    df_r_gen_std = df[df["out_channel"] == df["out_channel"].unique()[0]]
    fig = px.bar(df_r_gen_std, x="r_gen_index", y="std_r_gen", color="type", title="STD of r_gen", width=figsize[0],
                 height=figsize[1], barmode="group", log_y=log_y)
    if log_y:
        fig.update_layout(
            yaxis={
                'exponentformat': 'E'}
        )
    figs.append(fig)

    # r_gen times w_out figure
    for t in r_gen_data_types:
        df_temp = df[df["type"] == t]
        fig = px.bar(df_temp, x="r_gen_index", y="r_gen_std_times_w_out", color="out_channel", title=f"{t}: STD of r_gen times w_out", width=figsize[0],
                     height=figsize[1], barmode="group", log_y=log_y)
        if log_y:
            fig.update_layout(
                yaxis={
                    'exponentformat': 'E'}
            )
        figs.append(fig)
    return figs

def plot_histogram(hist_data, figsize=(650, 500), steps=-1, log_y=False):
    hist_data = {key: val[:steps, :].flatten() for key, val in hist_data.items()}
    df = pd.DataFrame()

    for key, val in hist_data.items():
        df_to_add = pd.DataFrame()
        df_to_add["data"] = val
        df_to_add["name"] = key
        df = pd.concat([df, df_to_add])

    # print(df.head())
    # print(df.shape)
    # print(df.describe())
    fig = px.histogram(df, x="data", color="name", histnorm='probability density', log_y=log_y)
    if log_y:
        fig.update_layout(
            yaxis={
                'exponentformat': 'E'}
        )
    fig.update_layout(height=figsize[1], width=figsize[0])
    return fig


def plot_valid_times_vs_pred_error(y_pred, y_true, error_thresh_min=0.1, error_thresh_max=1., steps=10, in_lyapunov_times=None):
    error_over_time = measures.error_over_time(y_pred, y_true, distance_measure="L2",
                                               normalization="root_of_avg_of_spacedist_squared")
    error_thresh_list = np.linspace(error_thresh_min, error_thresh_max, steps, endpoint=True)
    valid_times = np.zeros(steps)
    for i, thresh in enumerate(error_thresh_list):
        valid_time = measures.valid_time_index(error_over_time, thresh)
        valid_times[i] = valid_time

    if in_lyapunov_times is not None:
        dt = in_lyapunov_times["dt"]
        le = in_lyapunov_times["LE"]
        valid_times = dt*le*valid_times

    df = pd.DataFrame({"threshhold": error_thresh_list, "valid_times": valid_times})
    fig = px.line(df, x="threshhold", y="valid_times", title="valid_times vs error_threshhold")
    return fig


def plot_w_out_index_sweep(trajs, params_to_show, f, sweep_variable, figsize=(150, 150),
                           log_x=False, average_type="mean"):
    """
    for hdf5_viewer_wout.py
    """
    df = pd.DataFrame()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:].flatten()
        params = params_to_show[i_traj]

        if average_type == "mean":
            w_out_index_avg = np.mean(data)
            w_out_index_error_lower = np.std(data)
            w_out_index_error_upper = w_out_index_error_lower

        elif average_type == "median":
            w_out_index_avg = np.median(data)
            w_out_index_error_lower = w_out_index_avg - np.quantile(data, q=0.25)
            w_out_index_error_upper = np.quantile(data, q=0.75) - w_out_index_avg

        df_to_add = pd.DataFrame()
        df_to_add["w_out index"] = [w_out_index_avg]
        df_to_add["w_out index lower"] = [w_out_index_error_lower]
        df_to_add["w_out index upper"] = [w_out_index_error_upper]

        df_to_add[sweep_variable] = [params[sweep_variable]]

        params_str = ", ".join([f"{key}: {val}" for key, val in params.items() if key != sweep_variable])
        df_to_add["Other Parameters"] = [params_str]

        df = pd.concat([df, df_to_add])

    print(df)
    df.sort_values(["Other Parameters", sweep_variable], inplace=True)
    fig = px.line(df, x=sweep_variable, y="w_out index", error_y="w_out index upper",
                  error_y_minus="w_out index lower", color="Other Parameters", width=figsize[0],
                  height=figsize[1], log_x=log_x)

    if log_x:
        fig.update_layout(
            xaxis={
                'exponentformat': 'E'}
        )
    return fig


def plot_w_out_mean_distribution(data):
    """
    UNDER CONSTRUCTION and ugly coding (quick)
    """
    data = data[:, :-1]
    index = np.arange(data.shape[1])
    df = pd.DataFrame()
    df["avg_w_out_abs"] = np.mean(data, axis=0)
    df["std_w_out_abs"] = np.std(data, axis=0)
    df["mean_by_std_w_out_abs"] = df["avg_w_out_abs"]/df["std_w_out_abs"]
    df["median_w_out_abs"] = np.median(data, axis=0)
    df["index"] = index

    fig1 = px.bar(df, x="index", y="avg_w_out_abs")
    fig2 = px.bar(df, x="index", y="std_w_out_abs")
    fig3 = px.bar(df, x="index", y="median_w_out_abs")
    fig4 = px.bar(df, x="index", y="mean_by_std_w_out_abs")
    return fig1, fig2, fig3, fig4





def plot_1d_time_delay(time_series, i_dim=0, time_delay=1, line=True):
    time_steps = time_series.shape[0]
    time_series_new = np.zeros((time_steps - time_delay*3, 3))
    time_series_new[:, 0] = time_series[:-time_delay*3, i_dim]
    time_series_new[:, 1] = time_series[1:-time_delay*3+1, i_dim]
    time_series_new[:, 2] = time_series[2:-time_delay*3+2, i_dim]
    return plot_3d_time_series(time_series_new, line=line)
