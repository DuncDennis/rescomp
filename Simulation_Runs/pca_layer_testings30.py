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
    "t_pred": 800,
}

parameters = {
    "type": ["pca_layer", "normal_esn"],
    "r_dim_or_pca_comps": [100, 200, 300, 400, 500, 600, 700],
    "r_dim_for_pca": 700,
    "r_to_r_gen_opt": "output_bias",
    "act_fct_opt": "tanh",
    "node_bias_opt": "constant_bias",
    "bias_scale": 0.1,
    "reg_param": [1e-7],
    "w_in_opt": "ordered_sparse",
    "w_in_scale": [1.0],
    "n_rad": 0.5,
    "n_avg_deg": 20.0,
}


# DEFINE THE CREATE/TRAIN AND PREDICT FUNCTIONS
def model_creation_function(**kwargs):

    x_dim = 3
    if kwargs["type"] == "normal_esn":
        esn = ESN.ESN_normal()
        kwargs["r_dim"] = kwargs["r_dim_or_pca_comps"]
    elif kwargs["type"] == "pca_layer":
        kwargs["r_dim"] = kwargs["r_dim_for_pca"]
        kwargs["pca_components"] = kwargs["r_dim_or_pca_comps"]
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
    name = "22_06_2022_lorenz_rp_vs_p"
    seed = 1000
    N_ens = 10
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
