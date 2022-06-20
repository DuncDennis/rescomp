import numpy as np
import time
from . import utilities
from . import measures
from . import simulations
import os
import pathlib
import itertools
import matplotlib.pyplot as plt
import h5py
import pandas as pd


class StatisticalModelTester():
    '''
    A Class to statistically test one prediction model (rc or not),
    i.e. do an ensemble experiment
    '''

    def __init__(self):
        self.error_function = lambda y_pred, y_test: measures.error_over_time(y_pred, y_test, distance_measure="L2",
                                                                              normalization="root_of_avg_of_spacedist_squared")
        self.error_threshhold = None

        self.model_creation_function = lambda: None

        self._output_flag_synonyms = utilities._SynonymDict()
        self._output_flag_synonyms.add_synonyms(0, ["full"])
        self._output_flag_synonyms.add_synonyms(1, ["valid_times"])
        self._output_flag_synonyms.add_synonyms(2, ["valid_times_median_quartile"])
        self._output_flag_synonyms.add_synonyms(3, ["error"])
        self._output_flag = None

        self.results = None  # trajectories

    def set_error_function(self, error_function):
        self.error_function = error_function

    def set_model_creation_function(self, model_creation_function):
        '''
        :param model_creation_function: A function
        :return:
        '''
        self.model_creation_function = model_creation_function

    def set_model_prediction_function(self, model_prediction_function):
        '''
        :param model_prediction_function:
        :return:
        '''
        self.model_prediction_function = model_prediction_function

    def do_ens_experiment(self, nr_model_realizations, x_pred_list, output_flag="full", save_example_trajectory=False,
                          time_it=False, **kwargs):
        print("      Starting ensemble experiment...")
        print("      output_flag: ", output_flag)

        if time_it:
            t = time.time()

        self._output_flag = self._output_flag_synonyms.get_flag(output_flag)
        nr_of_time_intervals = len(x_pred_list)

        if self._output_flag in (1, 2):
            self.error_threshhold = kwargs["error_threshhold"]
            valid_times = np.zeros((nr_model_realizations, nr_of_time_intervals))

        for i in range(nr_model_realizations):
            print(f"Realization: {i + 1}/{nr_model_realizations} ...")
            model = self.model_creation_function()
            for j, x_pred in enumerate(x_pred_list):
                y_pred, y_test = self.model_prediction_function(x_pred, model)
                if self._output_flag == 0:
                    if i == 0 and j == 0:
                        predict_steps, dim = y_pred.shape
                        results = np.zeros((nr_model_realizations, nr_of_time_intervals, 2, predict_steps, dim))
                    results[i, j, 0, :, :] = y_pred
                    results[i, j, 1, :, :] = y_test
                elif self._output_flag in (1, 2):
                    valid_times[i, j] = measures.valid_time_index(self.error_function(y_pred, y_test),
                                                                  self.error_threshhold)
                elif self._output_flag == 3:
                    if i == 0 and j == 0:
                        errors = np.zeros((nr_model_realizations, nr_of_time_intervals, predict_steps))
                    errors[i, j, :] = self.error_function(y_pred, y_test)

        to_return = []

        if self._output_flag == 0:
            to_return.append(results)
        elif self._output_flag == 1:
            to_return.append(valid_times)
        elif self._output_flag == 2:
            median = np.median(valid_times)
            first_quartile = np.quantile(valid_times, 0.25)
            third_quartile = np.quantile(valid_times, 0.75)
            to_return.append(np.array([median, first_quartile, third_quartile]))
        elif self._output_flag == 3:
            to_return.append(errors)

        if time_it:
            elapsed_time = time.time() - t
            to_return.append(elapsed_time)
        if save_example_trajectory:
            example_trajectory = (y_pred, y_test)
            to_return.append(example_trajectory)
        return to_return

    def do_ens_experiment_internal(self, nr_model_realizations, x_pred_list, **kwargs):
        nr_of_time_intervals = len(x_pred_list)

        for i in range(nr_model_realizations):
            print(f"Realization: {i + 1}/{nr_model_realizations} ...")
            model = self.model_creation_function(**kwargs)
            for j, x_pred in enumerate(x_pred_list):
                y_pred, y_test = self.model_prediction_function(x_pred, model)
                if i == 0 and j == 0:
                    predict_steps, dim = y_pred.shape
                    results = np.zeros((nr_model_realizations, nr_of_time_intervals, 2, predict_steps, dim))
                results[i, j, 0, :, :] = y_pred
                results[i, j, 1, :, :] = y_test

        self.results = results

    def get_error(self, results=None, mean=False):
        if results is None:
            results = self.results

        n_ens = results.shape[0]
        n_interval = results.shape[1]
        n_pred_steps = results.shape[3]

        error = np.zeros((n_ens, n_interval, n_pred_steps))
        for i_ens in range(n_ens):
            for i_interval in range(n_interval):
                y_pred = results[i_ens, i_interval, 0, :, :]
                y_test = results[i_ens, i_interval, 1, :, :]

                error[i_ens, i_interval, :] = self.error_function(y_pred, y_test)
        if mean:
            error_mean = np.mean(error, axis=(0, 1))
            return error_mean
        return error

    def get_valid_times(self, error=None, results=None, mean=False, error_threshhold=None):
        if error is None:
            if results is None:
                results = self.results
            error = self.get_error(results)

        if error_threshhold is None:
            error_threshhold = self.error_threshhold

        n_ens = error.shape[0]
        n_interval = error.shape[1]

        valid_times = np.zeros((n_ens, n_interval))
        for i_ens in range(n_ens):
            for i_interval in range(n_interval):
                valid_times[i_ens, i_interval] = measures.valid_time_index(error[i_ens, i_interval, :],
                                                                           error_threshhold)

        if mean:
            valid_times_mean = np.mean(valid_times)
            return valid_times_mean
        return valid_times


class StatisticalModelTesterSweep(StatisticalModelTester):
    """

    """
    def __init__(self):
        super().__init__()

        self.input_parameters = None
        self.results_sweep = []
        self.error_sweep = []
        self.valid_times_sweep = []

        self.nr_model_realizations = None
        self.nr_of_time_intervals = None

    def _dict_of_vals_to_dict_of_list(self, inp):
        return {key: ((val,) if not type(val) in (list, tuple) else tuple(val)) for key, val in inp.items()}

    def _unpack_parameters(self, **parameters):
        list_of_params = []
        parameters_w_list = self._dict_of_vals_to_dict_of_list(parameters)

        keys = parameters_w_list.keys()
        values = parameters_w_list.values()

        for x in list(itertools.product(*values)):
            d = dict(zip(keys, x))
            list_of_params.append(d)

        return list_of_params

    def do_ens_experiment_sweep(self, nr_model_realizations, x_pred_list, results_type="trajectory", error_threshhold=0.4,
                                **parameters):
        # saves the whole trajectories or valid times (hopefully less memory consuming)
        self.input_parameters = parameters
        self.nr_model_realizations = nr_model_realizations
        self.nr_of_time_intervals = x_pred_list.shape[0]
        list_of_params = self._unpack_parameters(**parameters)
        for i, params in enumerate(list_of_params):
            print(f"Sweep: {i + 1}/{len(list_of_params)}")
            print(params)
            self.do_ens_experiment_internal(nr_model_realizations, x_pred_list, **params)

            if results_type == "trajectory":
                self.results_sweep.append((params, self.results.copy()))
            elif results_type == "validtimes":
                self.error_threshhold = error_threshhold
                vt = self.get_valid_times(results=self.results, error_threshhold=error_threshhold)
                self.valid_times_sweep.append((params, vt))

    def get_results_sweep(self):
        return self.results_sweep

    def get_error_sweep(self):
        # works if self.results_sweep is already populated
        error_sweep = []
        for params, results in self.results_sweep:
            error_sweep.append((params, self.get_error(results)))

        self.error_sweep = error_sweep
        return error_sweep

    def get_valid_times_sweep(self, error_threshhold=None):
        # works if self.results_sweep is already populated
        if error_threshhold is None:
            error_threshhold = self.error_threshhold

        else:
            self.error_threshhold = error_threshhold

        valid_times_sweep = []
        for params, results in self.results_sweep:
            vt = self.get_valid_times(results=results, error_threshhold=error_threshhold)
            valid_times_sweep.append((params, vt))

        self.valid_times_sweep = valid_times_sweep
        return valid_times_sweep

    def save_sweep_results(self, name="default_name", path=None, results_type="trajectory"):
        if path is None:
            repo_path = pathlib.Path(__file__).parent.resolve().parents[0]
            path = pathlib.Path.joinpath(repo_path, "results")
            print(path)

        if results_type == "trajectory":
            if len(self.results_sweep) == 0:
                raise Exception("no trajectory results yet")
            else:
                data = self.results_sweep
        elif results_type == "error":
            if len(self.error_sweep) == 0:
                raise Exception("no error results yet")
            else:
                data = self.error_sweep
        elif results_type == "validtimes":
            if len(self.valid_times_sweep) == 0:
                raise Exception("no valid_times results yet")
            else:
                data = self.valid_times_sweep

        # check if there is a file with that name already:
        if f"{name}.hdf5" in os.listdir(path):
            i = 0
            name_temp = name
            while f"{name_temp}.hdf5" in os.listdir(path):
                i += 1
                name_temp = f"{name}{i}"

            name = f"{name}{i}"
        print(name)
        file_path = pathlib.Path.joinpath(path, f"{name}.hdf5")
        print(file_path)
        with h5py.File(file_path, "w") as f:
            runs_group = f.create_group("runs")
            i = 1
            for params, results in data:
                dset = runs_group.create_dataset(f"trajectory_{i}", data=results)
                for key, val in params.items():
                    try:
                        dset.attrs[key] = val
                    except Exception as e:
                        print(e)
                        dset.attrs[key] = str(val)
                i += 1

            # sweep_info_group = f.create_group("sweep_info")
            # for key, val in self._dict_of_vals_to_dict_of_list(self.input_parameters).items():
            #     sweep_info_group.create_dataset(key, data=val)

    def get_valid_times_df(self, **kwargs):
        if len(self.valid_times_sweep) == 0 or "error_threshhold" in kwargs.keys():
            self.get_valid_times_sweep(**kwargs)

        df_vt = None

        for params, valid_times in self.valid_times_sweep:
            vt_mean = np.mean(valid_times)
            vt_std = np.std(valid_times)
            vt_median = np.median(valid_times)
            input = {key: (val, ) for key, val in params.items()}
            input["valid_times_mean"] = (vt_mean, )
            input["valid_times_median"] = (vt_median, )
            input["valid_times_std"] = (vt_std, )
            input["error_threshhold"] = (self.error_threshhold, )
            input["nr_model_realizations"] = (self.nr_model_realizations, )
            input["nr_of_time_intervals"] = (self.nr_of_time_intervals, )

            if df_vt is None:
                df_vt = pd.DataFrame.from_dict(input)

            else:
                df_vt = pd.concat([df_vt, pd.DataFrame.from_dict(input)])

        return df_vt

    def plot_error(self, ax=None, figsize=(15, 8)):
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        if len(self.error_sweep) == 0:
            self.get_error_sweep()

        for params, error in self.error_sweep:
            error_mean = np.mean(error, axis=(0, 1))
            ax.plot(error_mean, label=f"{params}")
        ax.legend()


class ST_sweeper():
    '''
    Older, new version is: StatisticalModelTesterSweep
    '''

    def __init__(self, sweeped_variable_dict, ST_creator_function, model_name="default_model_name",
                 saving_pre_path=None, artificial_sweep=False):
        self.sweeped_variable_name, self.sweeped_variable_list = list(sweeped_variable_dict.items())[0]
        self.ST_creator_function = ST_creator_function
        self.model_name = model_name
        self.output_flag = None
        self.pre_path = saving_pre_path  # for saving
        self.path = None
        if self.pre_path == None:
            self.saving = False
        else:
            self.saving = True
            self.check_path()
        self.artificial_sweep = artificial_sweep

    def check_path(self):
        pathlib.Path(self.pre_path).mkdir(parents=True, exist_ok=True)

    def sweep(self, **kwargs):
        print(f"STARTING SWEEP FOR MODEL: {self.model_name}")
        self.output_flag = kwargs["output_flag"]
        time_it = kwargs["time_it"]
        save_example_trajectory = kwargs["save_example_trajectory"]

        to_return = []

        results_sweeped = []

        if time_it:
            time_sweeped = []
        if save_example_trajectory:
            example_trajectory_sweeped = []

        for sweep_variable in self.sweeped_variable_list:
            print(f"{self.sweeped_variable_name}: {sweep_variable}")
            ST = self.ST_creator_function(sweep_variable)

            results_all = ST.do_ens_experiment(**kwargs)  # results can have multiple shapes

            if time_it:
                time = results_all[1]
                time_sweeped.append(time)
            if save_example_trajectory:
                example_trajectory = results_all[-1]
                example_trajectory_sweeped.append(example_trajectory)

            results_sweeped.append(results_all[0])

            if self.artificial_sweep:
                break
        if self.artificial_sweep:
            results_sweeped = results_sweeped * len(self.sweeped_variable_list)
            if time_it:
                time_sweeped = time_sweeped * len(self.sweeped_variable_list)
            if save_example_trajectory:
                example_trajectory_sweeped = example_trajectory_sweeped * len(self.sweeped_variable_list)

        results_sweeped = np.array(results_sweeped)
        to_return.append(results_sweeped)

        if time_it:
            time_sweeped = np.array(time_sweeped)
            to_return.append(time_sweeped)
        if save_example_trajectory:
            example_trajectory_sweeped = np.array(example_trajectory_sweeped)
            to_return.append(example_trajectory_sweeped)

        if self.saving:
            np.save(f"{self.pre_path}{self.model_name}__res__{self.output_flag}.npy", results_sweeped)
            np.save(f"{self.pre_path}{self.model_name}__sweep__{self.sweeped_variable_name}.npy",
                    self.sweeped_variable_list)
            if time_it:
                np.save(f"{self.pre_path}{self.model_name}__times.npy", time_sweeped)
            if save_example_trajectory:
                np.save(f"{self.pre_path}{self.model_name}__example_trajectory.npy", example_trajectory_sweeped)

        if len(to_return) == 1:
            return to_return[0]
        else:
            return to_return


def load_results(path):
    '''
    1) Check all files in the path
    :param path:
    :return:
    '''

    list_of_entries = os.listdir(path)
    list_of_files = [f for f in list_of_entries if os.path.isfile(os.path.join(path, f))]

    files_dict = {}
    for f in list_of_files:
        name_of_model = f.split("__")[0]
        if not name_of_model in files_dict.keys():
            files_dict[name_of_model] = [f, ]
        else:
            files_dict[name_of_model].append(f)
    results_bool = False
    time_it = False
    save_example_trajectory = False
    sweep_bool = False

    for key, val in files_dict.items():  # For each model
        for item in val:  # for each file of a model
            kind = item.split("__")[1]
            kind = kind.split(".")[0]
            if kind == "res":
                if not results_bool:
                    results_models = {}
                    output_flags_models = {}
                    results_bool = True
                results_models[key] = np.load(path + item)
                output_flag = item.split("__")[-1].split(".")[0]
                output_flags_models[key] = output_flag
            elif kind == "sweep":
                if not sweep_bool:
                    sweep_array_models = {}
                    sweep_bool = True
                sweep_array_models[key] = {}
                sweep_variable = item.split("__")[-1].split(".")[0]
                sweep_array_models[key][sweep_variable] = np.load(path + item)
            elif kind == "times":
                if not time_it:
                    times_models = {}
                    time_it = True
                times_models[key] = np.load(path + item)
            elif kind == "example_trajectory":
                if not save_example_trajectory:
                    example_trajectories_models = {}
                    save_example_trajectory = True
                example_trajectories_models[key] = np.load(path + item)

    to_return_dict = {}
    if results_bool:
        to_return_dict["results_models"] = results_models
        to_return_dict["output_flags_models"] = output_flags_models
    if time_it:
        to_return_dict["times_models"] = times_models
    if save_example_trajectory:
        to_return_dict["example_trajectories_models"] = example_trajectories_models
    if sweep_bool:
        to_return_dict["sweep_array_models"] = sweep_array_models
    return to_return_dict


def data_simulation(simulation_function_or_sim_data, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync,
                    t_pred, dt=1, nr_of_time_intervals=1, v=1, sim_data_return=False):
    """

    """
    train_disc_steps = int(t_train_disc / dt)
    train_sync_steps = int(t_train_sync / dt)
    train_steps = int(t_train / dt)
    pred_disc_steps = int(t_pred_disc / dt)
    pred_sync_steps = int(t_pred_sync / dt)
    pred_steps = int(t_pred / dt)
    total_time_steps = train_disc_steps + train_sync_steps + train_steps + (
            pred_disc_steps + pred_sync_steps + pred_steps) * nr_of_time_intervals

    if hasattr(simulation_function_or_sim_data, '__call__'):  # check if its a function:
        sim_data = simulation_function_or_sim_data(total_time_steps)
    else:
        sim_data = simulation_function_or_sim_data
        if sim_data.shape[0] < total_time_steps:
            raise Exception(f"total required time steps {total_time_steps} larger than input data!")

    x_train = sim_data[train_disc_steps: train_disc_steps + train_sync_steps + train_steps]

    x_pred_list = []
    start = train_disc_steps + train_sync_steps + train_steps - 1
    n_period = pred_disc_steps + pred_sync_steps + pred_steps
    for i in range(nr_of_time_intervals):
        # x_pred = sim_data[start + i * n_period + pred_disc_steps: start + (i + 1) * n_period + 1]
        x_pred = sim_data[start + i * n_period + pred_disc_steps: start + (i + 1) * n_period]
        x_pred_list.append(x_pred)
    x_pred_list = np.array(x_pred_list)

    if v == 1:
        print("train_disc_steps: ", train_disc_steps)
        print("train_sync_steps: ", train_sync_steps)
        print("train_steps: ", train_steps)
        print("pred_disc_steps: ", pred_disc_steps)
        print("pred_sync_steps: ", pred_sync_steps)
        print("pred_steps: ", pred_steps)
        print("total_time_steps: ", total_time_steps)
        print("................................")
        print("x_train shape: ", x_train.shape)
        print("x_pred_list shape :", x_pred_list.shape)

    if sim_data_return:
        print("sim_data shape :", sim_data.shape)
        return x_train, x_pred_list, sim_data

    return x_train, x_pred_list


def data_simulation_new(system="lorenz", t_train_disc=1000, t_train_sync=300, t_train=2000, t_pred_disc=1000,
                        t_pred_sync=300, t_pred=2000, nr_of_time_intervals=1, dt=0.05, normalize=False, train_noise=0.0,
                        sim_data_return=False, v=1, starting_point=None, t_settings_as_real_time=False):
    """

    """
    # t_train_disc = simulation_args["t_train_disc"]
    # t_train_sync = simulation_args["t_train_sync"]
    # t_train = simulation_args["t_train"]
    # t_pred_disc = simulation_args["t_pred_disc"]
    # t_pred_sync = simulation_args["t_pred_sync"]
    # t_pred = simulation_args["t_pred"]
    # nr_of_time_intervals = simulation_args["nr_of_time_intervals"]
    # dt = simulation_args["dt"]
    # system = simulation_args["system"]
    # normalize = simulation_args["normalize"]
    # train_noise = simulation_args["train_noise"]

    if t_settings_as_real_time:
        train_disc_steps = int(t_train_disc / dt)
        train_sync_steps = int(t_train_sync / dt)
        train_steps = int(t_train / dt)
        pred_disc_steps = int(t_pred_disc / dt)
        pred_sync_steps = int(t_pred_sync / dt)
        pred_steps = int(t_pred / dt)

    else:
        train_disc_steps = int(t_train_disc)
        train_sync_steps = int(t_train_sync)
        train_steps = int(t_train)
        pred_disc_steps = int(t_pred_disc)
        pred_sync_steps = int(t_pred_sync)
        pred_steps = int(t_pred)

    total_time_steps = train_disc_steps + train_sync_steps + train_steps + (
            pred_disc_steps + pred_sync_steps + pred_steps) * nr_of_time_intervals

    if starting_point is None:
        sim_data = simulations.simulate_trajectory(system, dt, total_time_steps)
    else:
        if starting_point == "standard":
            starting_point = simulations.standard_starting_points[system]

        sim_data = simulations.simulate_trajectory(system, dt, total_time_steps, starting_point)

    x_train = sim_data[train_disc_steps: train_disc_steps + train_sync_steps + train_steps]

    if normalize:
        sim_data = utilities.normalize_timeseries(sim_data, normalize_on=x_train)  # Only normalize on x train?
        x_train = sim_data[train_disc_steps: train_disc_steps + train_sync_steps + train_steps]

    # add noise
    x_train = x_train + np.random.randn(*(x_train.shape)) * train_noise

    x_pred_list = []
    start = train_disc_steps + train_sync_steps + train_steps - 1
    n_period = pred_disc_steps + pred_sync_steps + pred_steps
    for i in range(nr_of_time_intervals):
        x_pred = sim_data[start + i * n_period + pred_disc_steps: start + (i + 1) * n_period]
        x_pred_list.append(x_pred)
    x_pred_list = np.array(x_pred_list)

    if v == 1:
        print("train_disc_steps: ", train_disc_steps)
        print("train_sync_steps: ", train_sync_steps)
        print("train_steps: ", train_steps)
        print("pred_disc_steps: ", pred_disc_steps)
        print("pred_sync_steps: ", pred_sync_steps)
        print("pred_steps: ", pred_steps)
        print("total_time_steps: ", total_time_steps)
        print("................................")
        print("x_train shape: ", x_train.shape)
        print("x_pred_list shape :", x_pred_list.shape)

    if sim_data_return:
        print("sim_data shape :", sim_data.shape)
        return x_train, x_pred_list, sim_data

    return x_train, x_pred_list


# CUSTOM OVERWRITING OF CLASS FOR SPECIFIC TESTS:
class PCAWoutMeasurement(StatisticalModelTesterSweep):
    # Use with PCA esn to discover the mean of the w out distribution
    def __init__(self):
        super(PCAWoutMeasurement, self).__init__()

    def do_ens_experiment_internal(self, nr_model_realizations, x_pred_list, **kwargs):
        # nr_of_time_intervals = len(x_pred_list)

        for i in range(nr_model_realizations):
            print(f"Realization: {i + 1}/{nr_model_realizations} ...")
            model = self.model_creation_function(**kwargs)
            w_out = model._w_out

            w_out_abs = np.abs(w_out)
            w_out_summed = np.sum(w_out_abs, axis=0)
            w_out_summed = w_out_summed/np.sum(w_out_summed)
            mean_index = np.sum(w_out_summed*np.arange(1, len(w_out_summed)+1))
            if i == 0:
                results = np.zeros((nr_model_realizations))
            results[i] = mean_index

        self.results = results
