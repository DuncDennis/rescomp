import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# import pandas as pd
import math
import rescomp.measures as measures
import rescomp.utilities as utilities


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


def plot_correlation_dimension(trajs, params_to_show, f, i_ens, i_time_period, figsize=(5, 10), nr_steps=10):
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
def plot_3d_time_series(time_series):
    x = time_series[:, 0]
    y = time_series[:, 1]
    z = time_series[:, 2]
    fig = px.line_3d(x=x, y=y, z=z)
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


def plot_log_divergence(log_div_list, dt=1.0, fit=True, t_min=None, t_max=None, figsize=(9, 4), ax=None):
    time_steps = log_div_list.size
    t_list = np.arange(time_steps) * dt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    round_digs = 5

    if fit:
        if t_min is None:
            t_min = 0
        if t_max is None:
            t_max = t_list[-1]

        x_fit, y_fit, coefs = utilities._linear_fit(log_div_list, dt=dt, t_min=t_min, t_max=t_max)

        ax.plot(x_fit, y_fit,
                label=f"Sloap = {np.round(coefs[0], round_digs)}, Intersect = {np.round(coefs[1], round_digs)}",
                linestyle="--", c="k")

    ax.plot(t_list, log_div_list)
    ax.grid()
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(r"avg. log distance")


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


def plot_correlation_dimension(y_pred, y_true, nr_steps=20, figsize=(13, 5)):
    fig = plt.figure(figsize=figsize)

    sloap_true, N_r = measures.dimension(y_true, return_neighbours=True, nr_steps=nr_steps)
    plt.loglog(N_r[0], N_r[1], label=f"True: {sloap_true}")

    sloap_pred, N_r = measures.dimension(y_pred, return_neighbours=True, nr_steps=nr_steps)
    plt.loglog(N_r[0], N_r[1], label=f"Pred: {sloap_pred}")

    plt.legend()
    plt.xlabel("radius")
    plt.ylabel("Nr of Points")
    return fig


def plot_poincare_type_map(y_pred, y_true, dim=None, mode="maxima", figsize=(13, 5), s=1.0, alpha=1.0):
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
        x, y = measures.poincare_map(y_true, dimension=d, mode=mode)
        ax.scatter(x, y, label="True", s=s, alpha=alpha)

        x, y = measures.poincare_map(y_pred, dimension=d, mode=mode)
        ax.scatter(x, y, label="Pred", s=s, alpha=alpha)

        ax.set_title(f"Dimension: {d}")
        ax.legend()

    return fig


def plot_poincare_type_map_plotly(y_pred, y_true, dim=None, mode="maxima", figsize=(13, 5), s=1.0, alpha=1.0):
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
        x, y = measures.poincare_map(y_true, dimension=d, mode=mode)
        fig.add_trace(
            go.Scatter(x=x, y=y, opacity=alpha, name="True", mode='markers'),
            row=i_d+1, col=1
        )

        # ax.scatter(x, y, label="True", s=s, alpha=alpha)

        x, y = measures.poincare_map(y_pred, dimension=d, mode=mode)
        fig.add_trace(
            go.Scatter(x=x, y=y, opacity=alpha, name="Pred", mode='markers'),
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
