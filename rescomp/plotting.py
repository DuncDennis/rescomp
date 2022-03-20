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

def plot_error(trajs, params_to_show, f, max_x):
    fig = plt.figure()

    for i_traj, traj in enumerate(trajs):
        data = f["runs"][traj][:]
        error = get_error(data)
        mean_error = np.mean(error, axis=(0, 1))
        plt.plot(mean_error[:max_x], label=f"{params_to_show[i_traj]}")
    plt.legend()
    return fig


def plot_attractor_2(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(5, 10)):
    fig, axs = plt.subplots(len(trajs), 1, figsize=(base_fig_size[0], base_fig_size[1]*len(trajs)))

    for i_traj, traj in enumerate(trajs):
        ax = axs[i_traj]
        data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        y_pred = data[0, :, :]
        y_test = data[1, :, :]
        ax.plot(y_test[:, 0], y_test[:, 2], linewidth=1, label="test")
        ax.plot(y_pred[:, 0], y_pred[:, 2], linewidth=1, label="true")
        ax.set_title(f"{params_to_show[i_traj]}")
        ax.legend()
    return fig


def plot_trajectories(trajs, params_to_show, f, i_ens, i_time_period, base_fig_size=(5, 10)):
    fig, axs = plt.subplots(len(trajs), 1, figsize=(base_fig_size[0], base_fig_size[1]*len(trajs)))

    for i_traj, traj in enumerate(trajs):
        ax = axs[i_traj]
        data = f["runs"][traj][:][i_ens, i_time_period, :, :, :]
        y_pred = data[0, :, :]
        y_test = data[1, :, :]
        ax.plot(y_test[:, 0], label="test")
        ax.plot(y_pred[:, 0], label="true")
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
