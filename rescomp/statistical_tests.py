import numpy as np
from . import simulations, esn, utilities

class StatisticalModelTester():
    '''

    '''
    def __init__(self):
        self.error_function = None
        self.error_threshhold = None

        self.model_creation_function = None

        self._output_flag_synonyms = utilities._SynonymDict()
        self._output_flag_synonyms.add_synonyms(0, ["full"])
        self._output_flag_synonyms.add_synonyms(1, ["valid_times"])
        self._output_flag_synonyms.add_synonyms(2, ["valid_times_median_quartile"])
        self._output_flag_synonyms.add_synonyms(3, ["error"])
        self._output_flag = None

    def set_error_function(self, error_function):
        self.error_function = error_function

    def _get_valid_time_index(self, error_array):
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
        self.model_creation_function = model_creation_function

    def set_model_prediction_function(self, model_prediction_function):
        self.model_prediction_function = model_prediction_function

    def do_ens_experiment(self, nr_model_realizations, x_pred_list, output_flag = "full", **kwargs):
        print("Starting ensemble experiment...")
        print("output_flag: ", output_flag)

        self._output_flag = self._output_flag_synonyms.get_flag(output_flag)
        nr_of_time_intervals = len(x_pred_list)

        if self._output_flag in (1, 2):
            self.error_threshhold = kwargs["error_threshhold"]
            valid_times = np.zeros((nr_model_realizations, nr_of_time_intervals))

        for i in range(nr_model_realizations):
            print("=============================================")
            print(f"Realization: {i+1}/{nr_model_realizations} ..." )
            model = self.model_creation_function()
            for j, x_pred in enumerate(x_pred_list):
                y_pred, y_test = self.model_prediction_function(model, x_pred)
                if self._output_flag == 0:
                    if i == 0 and j == 0:
                        predict_steps, dim = y_pred.shape
                        results = np.zeros((nr_model_realizations, nr_of_time_intervals, 2, predict_steps, dim))
                    results[i, j, 0,  :, :] = y_pred
                    results[i, j, 1,  :, :] = y_test
                elif self._output_flag in (1,2):
                    valid_times[i, j] = self._get_valid_time_index(self.error_function(y_pred, y_test))
                elif self._output_flag == 3:
                    if i == 0 and j == 0:
                        errors = np.zeros((nr_model_realizations, nr_of_time_intervals, predict_steps))
                    errors[i,j, :] = self.error_function(y_pred, y_test)

        print("=============================================")
        if self._output_flag == 0:
            return results
        elif self._output_flag == 1:
            return valid_times
        elif self._output_flag == 2:
            median = np.median(valid_times)
            first_quartile = np.quantile(valid_times, 0.25)
            third_quartile = np.quantile(valid_times, 0.75)
            return median, first_quartile, third_quartile
        elif self._output_flag == 3:
            return errors

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
    x_train = sim_data[train_disc_steps :train_sync_steps + train_steps]

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
