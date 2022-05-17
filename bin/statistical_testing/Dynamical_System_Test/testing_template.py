import rescomp
import rescomp.esn_new_update_code as ESN
import rescomp.statistical_tests as st
import numpy as np

# CREATE SIMULATION DATA:
simulation_args = {
    "system": "lorenz",
    "dt": 0.05,
    "normalize": True,
    "nr_of_time_intervals": 20,
    "train_noise": 0.0,
    "t_train_disc": 1000,
    "t_train_sync": 300,
    "t_train": 2000,
    "t_pred_disc": 1000,
    "t_pred_sync": 300,
    "t_pred": 500,
}

# DEFINE EXPERIMENT PARAMETERS:
seed = 102
N_ens = 20

# parameters = {
#     "esntype": ["normal", "dynsys"],
#     "r_dim": [20, 25, 30, 35],
#     "r_to_r_gen_opt": "output_bias",
#     "act_fct_opt": "tanh",
#     "node_bias_opt": "constant_bias",
#     "bias_scale": 0.8,
#     "reg_param": 1e-7,
#     "w_in_opt": "random_dense_gaussian",
#     "w_in_scale": 0.05,
#     "dyn_sys_opt": "L96",
#     "dyn_sys_dt": 0.1,
#     "scale_factor": 1.0,
#     "L96_force": 0.0,
# }

parameters = {
    "esntype": ["normal"],
    "r_dim": [100],
    "r_to_r_gen_opt": "output_bias",
    "act_fct_opt": "tanh",
    "node_bias_opt": "constant_bias",
    "bias_scale": 0.8,
    "reg_param": 1e-7,
    "w_in_opt": "random_dense_gaussian",
    "w_in_scale": [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35],
    "n_rad": 0.5,
}


# DEFINE THE CREATE/TRAIN AND PREDICT FUNCTIONS
def model_creation_function(**kwargs):
    if kwargs["esntype"] == "dynsys":
        esn = ESN.ESN_dynsys()
    elif kwargs["esntype"] == "normal":
        esn = ESN.ESN_normal()
    else:
        raise Exception("esntype not recognized!")

    x_dim = 3

    del kwargs["esntype"]

    build_kwargs = rescomp.utilities._remove_invalid_args(esn.build, kwargs)

    esn.build(x_dim, **build_kwargs)
    esn.train(x_train, sync_steps=simulation_args["t_train_sync"])
    return esn


def model_prediction_function(x_pred, model):
    # returns y_pred and y_true
    return model.predict(x_pred, sync_steps=simulation_args["t_pred_sync"], save_r=False)


if __name__ == "__main__":
    print("Simulating Data")
    x_train, x_pred_list = st.data_simulation_new(**simulation_args, sim_data_return=False)

    print("Running Experiment")
    np.random.seed(seed)
    sweep_tester = st.StatisticalModelTesterSweep()
    sweep_tester.set_model_creation_function(model_creation_function)
    sweep_tester.set_model_prediction_function(model_prediction_function)
    sweep_tester.do_ens_experiment_sweep(N_ens, x_pred_list, **parameters)
    print("Experiment Finished!")

    sweep_tester.save_sweep_results(name="16_05_2022_TEST")
    print("SAVED!")
