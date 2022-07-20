"""Try to reproduce the plots of Pathak

This file: Reproduce Lorenz model valid_times vs reservoir dim
"""
import rescomp
import rescomp.esn_new_update_code as ESN
import rescomp.statistical_tests as st
import numpy as np
import yaml
from datetime import datetime

# CREATE SIMULATION DATA:
simulation_args = {
    "system": "lorenz",
    "dt": 0.1,
    "normalize": False,
    "nr_of_time_intervals": 20,
    "train_noise": 0.0,
    "t_train_disc": 5000,
    "t_train_sync": 100,
    "t_train": 1000,
    "t_pred_disc": 5000,
    "t_pred_sync": 100,
    "t_pred": 600,
}

parameters = {
    "type": ["full_hybrid", "normal", "only_model"],  # "output_hybrid", "input_hybrid",
    "eps": [0.004, 0.008, 0.012, 0.016, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    "r_dim": [50],
    "n_type_opt": ["random_directed"],
    "r_to_r_gen_opt": "linear_and_square_r_alt",
    "act_fct_opt": "tanh",
    "node_bias_opt": "no_bias",
    "bias_scale": 0.0,
    "reg_param": [1e-7, ],
    "w_in_opt": "random_sparse",
    "w_in_scale": [0.15],
    "n_rad": 0.4,
    "n_avg_deg": 3.0,
    "model_to_network_factor": 0.5,
}


# DEFINE THE CREATE/TRAIN AND PREDICT FUNCTIONS
def model_creation_function(**kwargs):

    x_dim = 3

    # sys_flag = simulation_args["system"]
    dt = simulation_args["dt"]

    modified_rho = 28*(1 + kwargs["eps"])

    model = rescomp.simulations_new.Lorenz63(rho=modified_rho, dt=dt).iterate

    if kwargs["type"] == "only_model":
        class model_class:
            def __init__(self):
                pass

            def predict(self, use_for_pred, sync_steps=0, save_r=False):
                if sync_steps > 0:
                    sync = use_for_pred[:sync_steps]
                    true_data = use_for_pred[sync_steps:]
                else:
                    true_data = use_for_pred

                steps = true_data.shape[0]
                prediction = np.zeros((steps, use_for_pred.shape[1]))
                prediction[0] = model(sync[-1])
                for i in range(1, steps):
                    prediction[i] = model(prediction[i-1])
                return prediction, true_data

        return model_class()

    else:
        if kwargs["type"] == "output_hybrid":
            esn = ESN.ESN_output_hybrid()
            kwargs["output_model"] = model
        elif kwargs["type"] == "input_hybrid":
            esn = ESN.ESN_input_hybrid()
            kwargs["input_model"] = model
        elif kwargs["type"] == "full_hybrid":
            esn = ESN.ESN_full_hybrid()
            kwargs["output_model"] = model
            kwargs["input_model"] = model
        elif kwargs["type"] == "output_hybrid_pca":
            esn = ESN.ESN_output_hybrid_pca()
            kwargs["output_model"] = model
        elif kwargs["type"] == "input_hybrid_pca":
            esn = ESN.ESN_input_hybrid_pca()
            kwargs["input_model"] = model
        elif kwargs["type"] == "full_hybrid_pca":
            esn = ESN.ESN_full_hybrid_pca()
            kwargs["output_model"] = model
            kwargs["input_model"] = model
        elif kwargs["type"] == "normal":
            esn = ESN.ESN_normal()

        build_kwargs = rescomp.utilities._remove_invalid_args(esn.build, kwargs)

        esn.build(x_dim, **build_kwargs)
        esn.train(x_train, sync_steps=simulation_args["t_train_sync"])
        return esn


def model_prediction_function(x_pred, model):
    # returns y_pred and y_true
    return model.predict(x_pred, sync_steps=simulation_args["t_pred_sync"], save_r=False)


def save_to_yaml(parameter_dict, name=""):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hh_%Mm_%Ss")
    f_name = name + "_" + dt_string
    with open(f'saved_parameters/{f_name}.yml', 'w') as outfile:
        yaml.dump(parameter_dict, outfile, default_flow_style=False)


if __name__ == "__main__":
    # DEFINE EXPERIMENT PARAMETERS:
    name = "18_07_2022_repr_pathak_lorenz_eps_sweep_2"
    seed = 1000
    N_ens = 32
    print("Simulating Data")
    x_train, x_pred_list = st.data_simulation_new(**simulation_args, sim_data_return=False, t_settings_as_real_time=False,
                                                  starting_point="standard")

    parameter_dict = {"sim_opts": simulation_args, "build_parameters": parameters, "seed": seed, "N_ens": N_ens}

    print("Running Experiment")
    np.random.seed(seed)
    sweep_tester = st.StatisticalModelTesterSweep()
    sweep_tester.set_model_creation_function(model_creation_function)
    sweep_tester.set_model_prediction_function(model_prediction_function)
    sweep_tester.do_ens_experiment_sweep(N_ens, x_pred_list, **{**parameters, **simulation_args})
    print("Experiment Finished!")

    sweep_tester.save_sweep_results(name=name)
    save_to_yaml(parameter_dict, name)

    print("SAVED!")
