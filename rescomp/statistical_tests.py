import numpy as np
import time
from pathlib import Path
from . import utilities
import os

class StatisticalModelTester():
    '''

    '''
    def __init__(self):
        self.error_function = None
        self.error_threshhold = None

        self.model_creation_function = lambda: None

        self._output_flag_synonyms = utilities._SynonymDict()
        self._output_flag_synonyms.add_synonyms(0, ["full"])
        self._output_flag_synonyms.add_synonyms(1, ["valid_times"])
        self._output_flag_synonyms.add_synonyms(2, ["valid_times_median_quartile"])
        self._output_flag_synonyms.add_synonyms(3, ["error"])
        self._output_flag = None

    def set_error_function(self, error_function):
        self.error_function = error_function

    def get_valid_time_index(self, error_array):
        '''
        :param error_array (np.ndarray):
        :return: (int) The timesteps where the error is bigger than self.error_threshhold
        '''
        f = self.error_threshhold
        return np.argmax(error_array>f)
        #
        # for i, error in enumerate(error_array):
        #     if error > f:
        #         return i
        # return i

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

    def do_ens_experiment(self, nr_model_realizations, x_pred_list, output_flag = "full", save_example_trajectory = False, time_it = False, **kwargs):
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
            print(f"Realization: {i+1}/{nr_model_realizations} ..." )
            model = self.model_creation_function()
            for j, x_pred in enumerate(x_pred_list):
                y_pred, y_test = self.model_prediction_function(x_pred, model)
                if self._output_flag == 0:
                    if i == 0 and j == 0:
                        predict_steps, dim = y_pred.shape
                        results = np.zeros((nr_model_realizations, nr_of_time_intervals, 2, predict_steps, dim))
                    results[i, j, 0,  :, :] = y_pred
                    results[i, j, 1,  :, :] = y_test
                elif self._output_flag in (1,2):
                    valid_times[i, j] = self.get_valid_time_index(self.error_function(y_pred, y_test))
                elif self._output_flag == 3:
                    if i == 0 and j == 0:
                        errors = np.zeros((nr_model_realizations, nr_of_time_intervals, predict_steps))
                    errors[i,j, :] = self.error_function(y_pred, y_test)

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

def data_simulation(simulation_function, t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred, dt, nr_of_time_intervals, v = 1, sim_data_return = False):
    train_disc_steps = int(t_train_disc / dt)
    train_sync_steps = int(t_train_sync / dt)
    train_steps = int(t_train / dt)
    pred_disc_steps = int(t_pred_disc / dt)
    pred_sync_steps = int(t_pred_sync / dt)
    pred_steps = int(t_pred / dt)
    total_time_steps = train_disc_steps + train_sync_steps + train_steps + (
                pred_disc_steps + pred_sync_steps + pred_steps) * nr_of_time_intervals

    sim_data = simulation_function(total_time_steps)
    x_train = sim_data[train_disc_steps : train_disc_steps + train_sync_steps + train_steps]

    x_pred_list = []
    start = train_disc_steps + train_sync_steps + train_steps
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

class ST_sweeper():
    '''
    '''
    def __init__(self, sweeped_variable_dict, ST_creator_function, model_name = "default_model_name", saving_pre_path = None, artificial_sweep = False):
        self.sweeped_variable_name, self.sweeped_variable_list = list(sweeped_variable_dict.items())[0]
        self.ST_creator_function = ST_creator_function
        self.model_name = model_name
        self.output_flag = None
        self.pre_path = saving_pre_path # for saving
        self.path = None
        if self.pre_path == None:
            self.saving = False
        else:
            self.saving = True
            self.check_path()
        self.artificial_sweep = artificial_sweep

    def check_path(self):
        Path(self.pre_path).mkdir(parents=True, exist_ok=True)

    # def set_saving(self, pre_path):
    #     self.pre_path = pre_path
    #     self.saving = True

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

            results_all = ST.do_ens_experiment(**kwargs) # results can have multiple shapes

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
            np.save(f"{self.pre_path}{self.model_name}__res__{self.output_flag}.npy" , results_sweeped)
            np.save(f"{self.pre_path}{self.model_name}__sweep__{self.sweeped_variable_name}.npy", self.sweeped_variable_list)
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
            files_dict[name_of_model] = [f,]
        else:
            files_dict[name_of_model].append(f)
    results_bool = False
    time_it = False
    save_example_trajectory = False
    sweep_bool = False

    for key, val in files_dict.items(): # For each model
        for item in val: # for each file of a model
            # print("item: ", item)
            kind = item.split("__")[1]
            kind = kind.split(".")[0]
            # print("kind: ", kind)
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
                sweep_array_models[key][sweep_variable] = np.load(path +item)
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
