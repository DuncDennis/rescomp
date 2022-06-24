import rescomp
import rescomp.esn_new_update_code as ESN
import rescomp.statistical_tests as st
import numpy as np
import yaml
from datetime import datetime

# CREATE SIMULATION DATA:
simulation_args = {
    "system": "lorenz",
    "dt": 0.05,
    "normalize": True,
    "nr_of_time_intervals": 10,
    "train_noise": 0.0,
    "t_train_disc": 1000,
    "t_train_sync": 300,
    "t_train": 2000,
    "t_pred_disc": 1000,
    "t_pred_sync": 300,
    "t_pred": 500,
}

parameters = {
    "type": ["pca_layer", "normal"],
    "r_dim": [400],
    "norm_with_expl_var": [True, False],
    "centering_pre_trans": [True, False],
    "r_to_r_gen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "constant_bias",
    "bias_scale": 0.1,
    "reg_param": [1e-14, 1e-12, 1e-10, 1e-8, 1e-6],
    "w_in_opt": "ordered_sparse",
    "w_in_scale": [1.0],
    "n_rad": 0.1,
    "n_avg_deg": 5.0,
}


# DEFINE THE CREATE/TRAIN AND PREDICT FUNCTIONS
def model_creation_function(**kwargs):

    x_dim = 3
    if kwargs["type"] == "normal":
        esn = ESN.ESN_normal()
    elif kwargs["type"] == "pca_layer":
        esn = ESN.ESN_pca()

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
    name = "14_06_2022_Lorenz_PCA_effects_reg_sweep"
    seed = 106
    N_ens = 20
    print("Simulating Data")
    x_train, x_pred_list = st.data_simulation_new(**simulation_args, sim_data_return=False, t_settings_as_real_time=False)

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
