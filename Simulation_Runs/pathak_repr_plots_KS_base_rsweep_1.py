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
    "system": "KS",
    "dt": 0.25,
    "normalize": False,
    "nr_of_time_intervals": 20,
    "train_noise": 0.0,
    "t_train_disc": 10000,
    "t_train_sync": 0,
    "t_train": 20000,
    "t_pred_disc": 10000,
    "t_pred_sync": 40,
    "t_pred": 750,
}

parameters = {
    "type": ["normal"],  # "output_hybrid", "input_hybrid",
    "eps": [1.0],
    "KS_dimensions": 64,
    "KS_L": 35,
    "r_dim": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000],
    # "r_dim": [300],
    "n_type_opt": ["random_directed"],
    "r_to_r_gen_opt": "linear_and_square_r_alt",
    "act_fct_opt": "tanh",
    "node_bias_opt": "no_bias",
    "bias_scale": 0.0,
    "reg_param": [1e-7, ],
    "w_in_opt": "random_sparse",
    "w_in_scale": [1.0],
    "n_rad": 0.4,
    "n_avg_deg": 3.0,
    "model_to_network_factor": 0.5,
}


# DEFINE THE CREATE/TRAIN AND PREDICT FUNCTIONS
def model_creation_function(**kwargs):

    dt = simulation_args["dt"]

    eps = kwargs["eps"]
    KS_L = kwargs["KS_L"]
    KS_dimensions = kwargs["KS_dimensions"]

    x_dim = KS_dimensions

    model = rescomp.simulations_new.KuramotoSivashinsky(eps=eps, dt=dt, dimensions=KS_dimensions, system_size=KS_L).iterate

    # modified_parameters = {"sigma": 10, "rho": 28 * (1 + kwargs["eps"]), "beta": 8/3}
    # model = lambda x: rescomp.simulations.simulate_trajectory(sys_flag=sys_flag, dt=dt, time_steps=2, starting_point=x,
    #                                                           )[-1, 0:-1]

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
    name = "18_07_2022_repr_pathak_KS_base_r_sweep_1"
    seed = 1001
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
