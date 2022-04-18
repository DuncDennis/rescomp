import matplotlib.pyplot as plt
import numpy as np
import rescomp.measures as measures


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
def plot_architecture(esn, figsize=(10, 5)):
    """
    Plot w_in distribution and plot
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    w_in = esn._w_in
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
