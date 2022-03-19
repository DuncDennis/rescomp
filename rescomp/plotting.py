import matplotlib.pyplot as plt
import numpy as np


def plot_attractor(out):
    N_ens = out.shape[0]
    nr_of_time_intervals = out.shape[1]
    y_pred_all = out[:, :, 0, :, :]
    y_test_all = out[:, :, 1, :, :]
    fig, axs = plt.subplots(N_ens, nr_of_time_intervals, figsize=(5*nr_of_time_intervals, 5*N_ens))

    # for i_ens in range(N_ens):
    #     for i_interval in range(nr_of_time_intervals):

    for i_ens in range(2):
        for i_interval in range(2):
            ax = axs[i_ens, i_interval]
            y_pred = y_pred_all[i_ens, i_interval, :, :]
            y_test = y_test_all[i_ens, i_interval, :, :]
            ax.plot(y_test[:, 0], y_test[:, 2], linewidth=1, label="true")
            ax.plot(y_pred[:, 0], y_pred[:, 2], linewidth=1, label="pred")
            ax.set_title(f"ENS: {i_ens}, Timeinterval: {i_interval}")
            ax.legend()

    return fig
