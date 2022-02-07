"""
Test different data creation functions
"""

import os
from datetime import date
import time
import yaml
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import re
import rescomp

def compare_simulations_leqr(create_sim_func, create_sp_func, create_sim_func_keys, create_sp_func_keys,
                    parameter_dict, save=False, prepath="", exp_name="", Nens=1, Nle=1, get_info=True,
                        prefix="leqr", traj_div=False, prefix_div="qrdiv"):
    '''
    Compare the Nle largest Lyapunov exponents for different simulations
    TODO: add proper docstring
    '''

    start = time.time()

    seperator = "--------------------"

    if save:
        if not exp_name == "":
            prepath = os.path.join(prepath, exp_name)
            os.makedirs(prepath)

    print(seperator)
    print("create_sim_func_keys", create_sim_func_keys)
    print("create_sp_func_keys", create_sp_func_keys)
    print(seperator)

    # make list out of non-iterable parameters
    for key, val in parameter_dict.items():
        if type(val) in (list, tuple):
            pass
        else:
            parameter_dict[key] = [val, ]

    # print parameters:
    print(seperator)
    print("PARAMETERS: ")
    for key, val in parameter_dict.items():
        if len(val) == 1:
            val = val[0]
        print(key, val)
    print(seperator)

    # to save results which are outputted:
    leqr_res_dict = {}
    if traj_div:
        traj_div_dict = {}

    # TODO: check if necessary keys are present in parameter_dict

    # function to perform cartesian product on a dict of lists
    def dict_product(inp):
        return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

    # function to return subset of dict, depending on list of keys:
    def get_subdict(d, keys=[], not_keys=[]):
        return {k: v for (k, v) in d.items() if ((k in keys) and (not k in not_keys))}

    # function to turn parameter to string
    def param_to_str(param):
        if type(param) == str:
            pass
        elif type(param) == int:
            param = str(param)
        elif type(param) == float:
            if (int(param) - param) == 0:
                param = str(int(param))
            else:
                param = "{:.2e}".format(param).replace(".", "p")
        return param

    # function to turn a parameter dict into a name for saving the file
    def param_dict_to_name(d):
        name = ""
        for key, val in d.items():
            param_to_str(val)
            if name == "":
                add = ""
            else:
                add = "__"
            name += f"{add}{key}_{val}"
        return name

    # create sub dict that only concern simulation and sp creation keys
    sim_and_sp_keys = create_sim_func_keys + create_sp_func_keys
    sim_and_sp_params_dict = get_subdict(parameter_dict, keys=sim_and_sp_keys)

    # calculate number of calculations needed:
    nr_calcs = 1
    for val in parameter_dict.values():
        nr_calcs *= len(val)

    print(f"STARTING CALCULATIONS ({nr_calcs}): ")
    c = 1  # counter for printing
    for sim_and_sp_dict in dict_product(sim_and_sp_params_dict):
        create_sim_func_params = get_subdict(sim_and_sp_dict, keys=create_sim_func_keys)
        create_sp_func_params = get_subdict(sim_and_sp_dict, keys=create_sp_func_keys)
        # now we have the parameters needed to calculate the data_creation function and the starting_points
        sim_func = create_sim_func(create_sim_func_params)
        starting_points = create_sp_func(create_sp_func_params, Nens)

        # iterate the starting points to after tau steps
        for dt in parameter_dict["dt"]:

            # define the iterator f for later
            def f(x):
                return sim_func(time_steps=2, dt=dt, starting_point=x)[-1]

            for tau in parameter_dict["tau"]:
                starting_points_after_tau = np.zeros(starting_points.shape)
                tau_steps = int(tau/dt)
                for iens in range(Nens):
                    starting_points_after_tau[iens, :] = sim_func(tau_steps, dt, starting_points[iens, :])[-1, :]

                for T in parameter_dict["T"]:
                    for eps in parameter_dict["eps"]:
                        for Nqr in parameter_dict["Nqr"]:
                            print("....")
                            print(f"CALC: {c}/{nr_calcs}")
                            print(f"{sim_and_sp_dict}, dt: {dt}, tau: {tau}, T: {T}, eps: {eps}, Nqr: {Nqr}")
                            out = rescomp.measures.iterator_based_lyapunov_spectrum(f,
                                                                     starting_points=starting_points_after_tau,
                                                                     T=T, nr_of_lyapunovs=Nle, nr_steps=Nqr,
                                                                     tau=0, dt=dt, eps=eps, jacobian=None,
                                                                     return_convergence=True,
                                                                     return_traj_divergence=traj_div,
                                                                                    agg=None)


                            out_leqr = out[1] # only get the convergence file
                            out_name = param_dict_to_name(sim_and_sp_dict)
                            out_name += f"__dt_{param_to_str(dt)}__tau_{param_to_str(tau)}__T_{param_to_str(T)}" \
                                    f"__eps_{param_to_str(eps)}__Nqr_{param_to_str(Nqr)}__Nle_{param_to_str(Nle)}" \
                                        f"__Nens_{param_to_str(Nens)}"

                            if prefix != "":
                                out_name_leqr = prefix + "__" + out_name

                            leqr_res_dict[out_name_leqr] = out_leqr

                            if save:
                                path = os.path.join(prepath, out_name_leqr)
                                np.save(path, out_leqr)

                            if traj_div:
                                out_traj_div = out[-1]
                                if prefix_div != "":
                                    out_name_div = prefix_div + "__" + out_name
                                traj_div_dict[out_name_div] = out_traj_div
                                if save:
                                    path = os.path.join(prepath, out_name_div)
                                    np.save(path, out_traj_div)

                            c += 1

    # aggregate data in a info_dict and return (or save) it
    if get_info:
        info_dict = {}
        info_dict["exp_name"] = exp_name
        info_dict["prefix"] = prefix
        info_dict["traj_div"] = traj_div
        if traj_div:
            info_dict["prefix_div"] = prefix_div
        info_dict["parameter_dict"] = parameter_dict
        info_dict["Nens"] = Nens
        info_dict["Nle"] = Nle
        info_dict["date"] = date.today().strftime("%d/%m/%Y")
        info_dict["time[min]"] = str(round((time.time() - start)/60, 3))
        if save:
            info_name = "info"
            if prefix != "":
                info_name = prefix + "__" + info_name
            if not exp_name == "":
                info_name = info_name + "_" + exp_name + ".yml"
                path = os.path.join(prepath, info_name)
                with open(path, 'w') as outfile:
                    yaml.dump(info_dict, outfile, default_flow_style=False)

    to_return = [leqr_res_dict, ]
    if get_info:
        to_return.append(info_dict)
    if traj_div:
        to_return.append(traj_div_dict)

    if len(to_return) == 1:
        to_return = to_return[0]

    return to_return


def compare_simulations_div(create_sim_func, create_sp_func, create_sim_func_keys, create_sp_func_keys,
                    parameter_dict, save=False, prepath="", exp_name="", Nens=1, Ndims=1, get_info=True,
                        prefix="div", random_directions=False):
    '''
    Compare the divergence ln(d(t)/d_0) for differenent simulations
    TODO: add proper docstring
    '''

    start = time.time()

    seperator = "--------------------"

    if save:
        if not exp_name == "":
            prepath = os.path.join(prepath, exp_name)
            os.makedirs(prepath)

    print(seperator)
    print("create_sim_func_keys", create_sim_func_keys)
    print("create_sp_func_keys", create_sp_func_keys)
    print(seperator)

    # make list out of non-iterable parameters
    for key, val in parameter_dict.items():
        if type(val) in (list, tuple):
            pass
        else:
            parameter_dict[key] = [val, ]

    # print parameters:
    print(seperator)
    print("PARAMETERS: ")
    for key, val in parameter_dict.items():
        if len(val) == 1:
            val = val[0]
        print(key, val)
    print(seperator)

    # to save results which are outputted:
    div_res_dict = {}

    # TODO: check if necessary keys are present in parameter_dict

    # function to perform cartesian product on a dict of lists
    def dict_product(inp):
        return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

    # function to return subset of dict, depending on list of keys:
    def get_subdict(d, keys=[], not_keys=[]):
        return {k: v for (k, v) in d.items() if ((k in keys) and (not k in not_keys))}

    # function to turn parameter to string
    def param_to_str(param):
        if type(param) == str:
            pass
        elif type(param) == int:
            param = str(param)
        elif type(param) == float:
            if (int(param) - param) == 0:
                param = str(int(param))
            else:
                param = "{:.2e}".format(param).replace(".", "p")
        return param

    # function to turn a parameter dict into a name for saving the file
    def param_dict_to_name(d):
        name = ""
        for key, val in d.items():
            param_to_str(val)
            if name == "":
                add = ""
            else:
                add = "__"
            name += f"{add}{key}_{val}"
        return name

    # create sub dict that only concern simulation and sp creation keys
    sim_and_sp_keys = create_sim_func_keys + create_sp_func_keys
    sim_and_sp_params_dict = get_subdict(parameter_dict, keys=sim_and_sp_keys)

    # calculate number of calculations needed:
    nr_calcs = 1
    for val in parameter_dict.values():
        nr_calcs *= len(val)

    print(f"STARTING CALCULATIONS ({nr_calcs}): ")
    c = 1  # counter for printing
    for sim_and_sp_dict in dict_product(sim_and_sp_params_dict):
        # print("sim_and_sp_dict: ", sim_and_sp_dict)
        create_sim_func_params = get_subdict(sim_and_sp_dict, keys=create_sim_func_keys)
        create_sp_func_params = get_subdict(sim_and_sp_dict, keys=create_sp_func_keys)
        # now we have the parameters needed to calculate the data_creation function and the starting_points
        sim_func = create_sim_func(create_sim_func_params)
        starting_points = create_sp_func(create_sp_func_params, Nens)

        # iterate the starting points to after tau steps
        for dt in parameter_dict["dt"]:

            # define the iterator f for later
            def f(x):
                return sim_func(time_steps=2, dt=dt, starting_point=x)[-1]

            for tau in parameter_dict["tau"]:
                print("simulate the tau time steps ...")
                starting_points_after_tau = np.zeros(starting_points.shape)
                tau_steps = int(tau/dt)
                for iens in range(Nens):
                    starting_points_after_tau[iens, :] = sim_func(tau_steps, dt, starting_points[iens, :])[-1, :]

                for T in parameter_dict["T"]:
                    for eps in parameter_dict["eps"]:
                        print("....")
                        print(f"CALC: {c}/{nr_calcs}")
                        print(f"{sim_and_sp_dict}, dt: {dt}, tau: {tau}, T: {T}, eps: {eps}")
                        out = rescomp.measures.calculate_divergence(f, starting_points=starting_points_after_tau,
                                                                 T=T,
                                                                 tau=0, dt=dt, eps=eps, N_dims=Ndims,
                                                                 agg=None, random_directions=random_directions)

                        out_name = param_dict_to_name(sim_and_sp_dict)
                        out_name += f"__dt_{param_to_str(dt)}__tau_{param_to_str(tau)}__T_{param_to_str(T)}" \
                                f"__eps_{param_to_str(eps)}__Ndims_{param_to_str(Ndims)}__Nens_{param_to_str(Nens)}"

                        if prefix != "":
                            out_name = prefix + "__" + out_name

                        div_res_dict[out_name] = out

                        if save:
                            path = os.path.join(prepath, out_name)
                            np.save(path, out)
                        c += 1

    # aggregate data in a info_dict and return (or save) it
    if get_info:
        info_dict = {}
        info_dict["exp_name"] = exp_name
        info_dict["prefix"] = prefix
        info_dict["parameter_dict"] = parameter_dict
        info_dict["Nens"] = Nens
        info_dict["Ndims"] = Ndims
        info_dict["date"] = date.today().strftime("%d/%m/%Y")
        info_dict["time[min]"] = str(round((time.time() - start)/60, 3))
        if save:
            info_name = "info"
            if prefix != "":
                info_name = prefix + "__" + info_name
            if not exp_name == "":
                info_name = info_name + "_" + exp_name + ".yml"
                path = os.path.join(prepath, info_name)
                with open(path, 'w') as outfile:
                    yaml.dump(info_dict, outfile, default_flow_style=False)

    if get_info:
        return [info_dict, div_res_dict]
    else:
        return div_res_dict

def load_info_file(prepath, exp_name):
    path = os.path.join(prepath, exp_name)
    files = os.listdir(path)
    # info_file_list = [x for x in files if x.startswith("info")]
    info_file_list = [x for x in files if (("info" in x) and ("yml" in x))]
    if len(info_file_list) == 0:
        raise Exception(f"(..)info(..).yml file not found in {path}")
    info_file = info_file_list[0]
    info_path = os.path.join(path,info_file)
    with open(info_path, 'r') as f:
        info_dict = yaml.load(f, Loader=yaml.Loader)
    return info_dict

def plot_div_file(ax, file_name, prepath="", exp_name="", mean_axs="all", dim_index=None,
                  ens_index=None, show_error=True, title=True,
                  rel_dist=True, include_fit=True, t_min=0, t_max=5, label="", verb=1,
                  include_fit_qr=False, t_min_qr=0, t_max_qr=5):
    '''
    Args:
        ax:
        file_name:
        prepath:
        exp_name:
        mean_axs: "no", "dim", "ens", "all"
        show_error:
        rel_dist:
        include_fit:
    Returns:
    '''

    path = os.path.join(prepath, exp_name)
    files = os.listdir(path)

    file_list = [x for x in files if x.startswith(file_name)]
    if len(file_list) == 0:
        raise Exception(f"no file found under {path}, that starts with {file_name}")
    elif len(file_list) > 1:
        raise Exception(f"multiple files found under {path}, that starts with {file_name}")
    else:
        file = file_list[0]

    params = file_name_to_params(file_name)
    T = params["T"]
    dt = params["dt"]

    array = np.load(os.path.join(path, file))
    array_shape = array.shape
    if verb == 1:
        print(file_name)
        print(f"shape of file: {array_shape}")

    if rel_dist:
        array = array/array[0, :, :]

    # turn to logarithmic values: (has to be done before calculating means
    array = np.log(array)

    if mean_axs == "no":
        dist = array
        error = np.zeros(array_shape)
    elif mean_axs == "dim":
        dist = np.mean(array, axis=-2)[:, np.newaxis, :]
        error = np.std(array, axis=-2)[:, np.newaxis, :]
    elif mean_axs == "ens":
        dist = np.mean(array, axis=-1)[:, :, np.newaxis]
        error = np.std(array, axis=-1)[:, :, np.newaxis]
    elif mean_axs == "all":
        dist = np.mean(array, axis=(-1, -2))[:, np.newaxis, np.newaxis]
        error = np.std(array, axis=(-1, -2))[:, np.newaxis, np.newaxis]
    else:
        raise Exception(f"Mean_axs {mean_axs} is not accounted for!")

    x_steps, n_dims_mod, n_ens_mod = dist.shape

    if not show_error:
        error = np.zeros(dist.shape)

    if dim_index is None:
        dim_index = np.arange(n_dims_mod)
    elif type(dim_index) == int:
        dim_index = [dim_index, ]
    if ens_index is None:
        ens_index = np.arange(n_ens_mod)
    elif type(ens_index) == int:
        ens_index = [ens_index, ]
    # print("x_steps: ", x_steps)
    # print("T: ", T)
    x = np.arange(x_steps)*dt
    for i_dim in range(n_dims_mod):
        for i_ens in range(n_ens_mod):
            if i_dim in dim_index and i_ens in ens_index:
                data = dist[:, i_dim, i_ens]
                data_error = error[:, i_dim, i_ens]
                data_error_high, data_error_low = data + data_error, data - data_error

                # # transform to logarithmic
                # data = np.log(data)
                # data_error_high = np.log(data_error_high)
                # data_error_low = np.log(data_error_low)

                label_current = label
                # linear fit
                if include_fit:
                    x_fit, y_fit, coef = linear_fit(data, dt, t_min=t_min, t_max=t_max)
                    if label_current == "":
                        add = ""
                    else:
                        add = ", "
                    label_current = label_current + f"{add}LLE: {coef[0].round(3)}"

                # connection fit vs first and last value
                if include_fit_qr:
                    x_fit_fl, y_fit_fl, coef_fl = connection_fit(data, dt, t_min=t_min_qr, t_max=t_max_qr)
                    if label_current == "":
                        add = ""
                    else:
                        add = ", "

                    label_current = label_current + f"{add}firstlast: {coef_fl[0].round(3)}"

                p = ax.errorbar(x=x, y=data, yerr=(data_error_high - data, data - data_error_low),
                                # label=label_current,
                                alpha=0.2)
                c = p[0].get_color()
                ax.plot(x, data, c=c, label=label_current)
                if include_fit:
                    ax.plot(x_fit, y_fit, linestyle="--", c=c)

                if include_fit_qr:
                    ax.plot(x_fit_fl, y_fit_fl, linestyle=":", c=c)

    ax.legend(loc='upper left')
    ax.grid(True)
    if title:
        ax.set_title(", ".join([f"{key}: {val}" for key,val in params.items()]))

    ax.set_ylabel(r"$\ln d/d_0$")
    ax.set_xlabel("T")

def plot_qrdiv_file(ax, file_name, prepath="", exp_name="", mean_axs="all", le_index=None,
                  ens_index=None, show_error=True, title=True, rel_dist=True,
                  label="", verb=1,
                  include_fit_qr=False):
    '''
    Args:
        ax:
        file_name:
        prepath:
        exp_name:
        mean_axs: "no", "dim", "ens", "all"
        show_error:
        rel_dist:
        include_fit:
    Returns:
    '''

    path = os.path.join(prepath, exp_name)
    files = os.listdir(path)

    file_list = [x for x in files if x.startswith(file_name)]
    if len(file_list) == 0:
        raise Exception(f"no file found under {path}, that starts with {file_name}")
    elif len(file_list) > 1:
        raise Exception(f"multiple files found under {path}, that starts with {file_name}")
    else:
        file = file_list[0]

    params = file_name_to_params(file_name)
    T = params["T"]
    Nqr = params["Nqr"]
    dt = params["dt"]

    array = np.load(os.path.join(path, file))
    array_shape = array.shape
    if verb == 1:
        print(file_name)
        print(f"shape of file: {array_shape}")

    if rel_dist:
        array = array/((array[:, :, :, 0])[:, :, :, np.newaxis])

    # turn to logarithmic:
    array = np.log(array)

    if mean_axs == "no":
        dist = array
        error = np.zeros(array_shape)
    elif mean_axs == "ens":
        dist = np.mean(array, axis=0)[np.newaxis, :, :, :]
        error = np.std(array, axis=0)[np.newaxis, :, :, :]
    else:
        raise Exception(f"Mean_axs {mean_axs} is not accounted for!")

    n_ens_mod, _, Nle, x_steps = dist.shape

    if not show_error:
        error = np.zeros(dist.shape)

    if ens_index is None:
        ens_index = np.arange(n_ens_mod)
    elif type(ens_index) == int:
        ens_index = [ens_index, ]

    if le_index is None:
        le_index = np.arange(Nle)
    elif type(le_index) == int:
        le_index = [le_index, ]

    x = np.arange(x_steps)
    if include_fit_qr:
        ax_sloap = ax.twinx()
        ax_sloap.set_ylabel("LEs for this qr step")
    for i_le in le_index:
        for i_ens in ens_index:
            if include_fit_qr:
                sloaps = np.zeros(Nqr)

            for i_Nqr in range(Nqr):
                ax.axvline((i_Nqr + 1) * T, c="k", linestyle=":")
                x_current = (x + i_Nqr * (x_steps - 1)) * dt

                data = dist[i_ens, i_Nqr, i_le, :]
                data_error = error[i_ens, i_Nqr, i_le, :]

                data_error_high, data_error_low = data + data_error, data - data_error

                # # transform to logarithmic
                # data = np.log(data)
                # data_error_high = np.log(data_error_high)
                # data_error_low = np.log(data_error_low)

                label_current = label

                p = ax.errorbar(x=x_current, y=data, yerr=(data_error_high - data, data - data_error_low),
                                label=label_current, alpha=0.1)
                c = p[0].get_color()
                ax.plot(x_current, data, c=c)

                if include_fit_qr:
                    t_min_qr = x_current[0]
                    t_max_qr = x_current[-1]
                    sloap = (data[-1] - data[0])/(t_max_qr - t_min_qr)
                    ax.plot((t_min_qr, t_max_qr), (data[0], data[-1]), linestyle="--", c=c)
                    sloaps[i_Nqr] = sloap
            if include_fit_qr:

                x_sloap = (np.arange(Nqr)+1)*T
                ax_sloap.plot(x_sloap, sloaps, '--o', c="k")
                ax_sloap.axhline(np.mean(sloaps), c = "r", linestyle=":")
                print(f"mean sloap: {np.mean(sloaps).round(3)}")

    ax.legend(loc='upper left')
    ax.grid(True)
    if title:
        ax.set_title(", ".join([f"{key}: {val}" for key,val in params.items()]))

    ax.set_ylabel(r"$\ln d/d_0$")
    ax.set_xlabel("Time")


def plot_leqr_file(ax, file_name, prepath="", exp_name="", mean_axs="all", le_index=None,
                  ens_index=None, show_error=True, title=True,
                  label="", verb=1, every_x_tick=False):
    '''
    Args:
        ax:
        file_name:
        prepath:
        exp_name:
        mean_axs: "no", "dim", "ens", "all"
        show_error:
        rel_dist:
        include_fit:
    Returns:
    '''

    path = os.path.join(prepath, exp_name)
    files = os.listdir(path)

    file_list = [x for x in files if x.startswith(file_name)]
    if len(file_list) == 0:
        raise Exception(f"no file found under {path}, that starts with {file_name}")
    elif len(file_list) > 1:
        raise Exception(f"multiple files found under {path}, that starts with {file_name}")
    else:
        file = file_list[0]

    params = file_name_to_params(file_name)
    Nqr = params["Nqr"]

    array = np.load(os.path.join(path, file))
    array_shape = array.shape
    if verb == 1:
        print(file_name)
        print(f"shape of file: {array_shape}")

    if mean_axs == "no":
        les = array
        error = np.zeros(array_shape)
    elif mean_axs == "ens":
        les = np.mean(array, axis=0)[np.newaxis, :, :]
        error = np.std(array, axis=0)[np.newaxis, :, :]
    else:
        raise Exception(f"Mean_axs {mean_axs} is not accounted for!")

    n_ens_mod, x_steps, Nle = les.shape

    if not show_error:
        error = np.zeros(les.shape)

    if ens_index is None:
        ens_index = np.arange(n_ens_mod)
    elif type(ens_index) == int:
        ens_index = [ens_index, ]

    if le_index is None:
        le_index = np.arange(Nle)
    elif type(le_index) == int:
        le_index = [le_index, ]

    x = np.arange(x_steps)
    for i_le in le_index:
        for i_ens in ens_index:
            data = les[i_ens, :, i_le]
            data_error = error[i_ens, :, i_le]

            label_current = label
            p = ax.errorbar(x=x, y=data, yerr=data_error, label=label_current, marker="x") #

    print(f"latest LEs: {np.mean(les[:, -1, :], axis=0).round(3)}")
    if every_x_tick:
        ax.set_xticks(x)
    ax.legend(loc='upper left')
    ax.grid(True)
    if title:
        ax.set_title(", ".join([f"{key}: {val}" for key, val in params.items()]))

    ax.set_ylabel(r"LEs")
    ax.set_xlabel("Nqr")

def file_name_to_params(file_name):

    if "." in file_name:
        file_name = file_name.split(".")[0]

    params = file_name.split("__")
    if not "_" in params[0]: # check for prefix-> prefix doesnt have "_" in name
        params = params[1:]

    params_keys = [x.split("_")[0] for x in params]
    params_vals = [x.split("_")[1] for x in params]

    params = dict([(re.search("\D+", key).group(), val) for (key, val) in zip(params_keys, params_vals)])

    for key, val in params.items():
        if re.search("[0-9]p", val) is not None: # floating point values
            params[key] = float(params[key].replace("p", "."))
        elif re.search("\D", val) is None: # other numerical values
            params[key] = int(params[key])

    return params

def linear_fit(y, dt, t_min=0, t_max=5):
    i_min, i_max = int(t_min/dt), int(t_max/dt)
    y = y[i_min: i_max+1]
    x_fit = np.arange(i_min, i_max+1)*dt
    coef = np.polyfit(x_fit,y,1)
    poly1d_fn = np.poly1d(coef)
    y_fit = poly1d_fn(x_fit)
    return x_fit, y_fit, coef

def connection_fit(y, dt, t_min=0, t_max=5):
    i_min, i_max = int(t_min / dt), int(t_max / dt)
    y = y[i_min: i_max + 1]
    x_fit = np.arange(i_min, i_max + 1) * dt

    x_fit_first_last = [x_fit[0], x_fit[-1]]
    y_first_last = [y[0], y[-1]]

    coef = np.polyfit(x_fit_first_last, y_first_last, 1)
    poly1d_fn = np.poly1d(coef)
    y_fit = poly1d_fn(x_fit)
    return x_fit, y_fit, coef


def plot_experiment(plot_func, prepath="", exp_name="", prefix="", zs=[], xs=[], ys=[], subfigsize=(10, 10),
                    xlims=None, ylims=None , **kwargs):
    '''
    Args:
        params_dict: a dictionary of the kind {"param1": [param1_val1, param1_val2], "param2": ...}
        data: a list of the kind: [(specific_parameter_dict_1, data_1), (..), ..].
            specific_parameter_dict_1 = {"param1": param1_val2, "param2": param2_val1, ..}
            data_1 = some kind of array or similar that is used by plot_func
        plot_func: takes axis as input and plots onto it using also specific_parameter_dict_1, data_1 as input
    Returns:
        figure (?)
    '''
    fig_size_x, fig_size_y = subfigsize

    if type(xs) == str:
        xs = [xs, ]
    if type(ys) == str:
        ys = [ys, ]
    if type(zs) == str:
        zs = [zs, ]

    info_dict = load_info_file(prepath, exp_name)

    data_files = os.listdir(os.path.join(prepath, exp_name))
    data_files = [x for x in data_files if not "info" in x]
    if prefix != "":
        data_files = [x for x in data_files if x.startswith(prefix+"__")]
    list_of_params_dict = [file_name_to_params(x) for x in data_files] # match the parameters

    params_dict = info_dict["parameter_dict"]

    variables_sweeped = {key: len(val) for (key, val) in params_dict.items()}

    nr_cols = 1
    for x in [variables_sweeped[x] for x in xs]:
        nr_cols *= x

    nr_rows = 1
    for y in [variables_sweeped[y] for y in ys]:
        nr_rows *= y

    nr_lines = 1
    for z in [variables_sweeped[z] for z in zs]:
        nr_lines *= z

    print(f"lines: {nr_lines}, cols: {nr_cols}, rows: {nr_rows}")

    i = 0
    x_index_to_sweep = {}
    x_arg_list = [params_dict[x] for x in xs]
    for items in product(*x_arg_list):
        x_index_to_sweep[i] = items
        i += 1

    i = 0
    y_index_to_sweep = {}
    y_arg_list = [params_dict[y] for y in ys]
    for items in product(*y_arg_list):
        y_index_to_sweep[i] = items
        i += 1

    i = 0
    z_index_to_sweep = {}
    z_arg_list = [params_dict[z] for z in zs]
    for items in product(*z_arg_list):
        z_index_to_sweep[i] = items
        i += 1

    fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(fig_size_x*nr_cols, fig_size_y*nr_rows),
                            )
    fig.patch.set_facecolor('white')
    # fig.patch.set_alpha(0.6)

    for i_x in range(nr_cols):
        for i_y in range(nr_rows):
            if nr_cols == 1:
                if nr_rows == 1:
                    ax = axs
                else:
                    ax = axs[i_y]
            else:
                if nr_rows == 1:
                    ax = axs[i_x]
                else:
                    ax = axs[i_y, i_x]

            x_params = x_index_to_sweep[i_x]
            y_params = y_index_to_sweep[i_y]

            x_params_key = dict(list(zip(xs, x_params)))
            y_params_key = dict(list(zip(ys, y_params)))

            xy_params_key = {**x_params_key, **y_params_key}

            for i_z in range(nr_lines):
                z_params = z_index_to_sweep[i_z]
                z_params_key = dict(list(zip(zs, z_params)))
                all_params_key = {**xy_params_key, **z_params_key}

                ix = match_params(list_of_params_dict, all_params_key)
                file_name = data_files[ix]
                label = ", ".join([f"{key}: {val}" for key, val in z_params_key.items()])

                plot_func(ax, file_name, prepath=prepath, exp_name=exp_name, verb=0, title=False, label=label, **kwargs)
                # plot_div_file(ax, file_name, prepath=prepath, exp_name=exp_name, verb=0, title=False, label=label, **kwargs)
                title_string = ", ".join([f"{key}: {val}" for key, val in xy_params_key.items()])
                ax.set_title(title_string)
                if xlims is not None:
                    ax.set_xlim(xlims[0], xlims[1])
                if ylims is not None:
                    ax.set_ylim(ylims[0], ylims[1])
    # plot suptitle:
    keys_to_avoid = list(xs) + list(ys) + list(zs)
    bigtitle = ", ".join([f"{k}: {v}" for k, v in params_dict.items() if k not in keys_to_avoid])
    fig.suptitle(bigtitle, fontsize="x-large")


def match_params(list_of_params_dict, params_dict):
    for ix, x in enumerate(list_of_params_dict):
        subdict = {k: v for (k, v) in x.items() if k in params_dict.keys()}
        if subdict == params_dict:
            return ix
