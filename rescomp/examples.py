import numpy as np
import rescomp.esn_new_update_code as ESN
import rescomp.statistical_tests as st

def trained_normal_esn():
    simulation_args = {
        "system": "lorenz",
        "dt": 0.05,
        "normalize": True,
        "nr_of_time_intervals": 1,
        "train_noise": 0.0,
        "t_train_disc": 1000,
        "t_train_sync": 300,
        "t_train": 2000,
        "t_pred_disc": 1000,
        "t_pred_sync": 300,
        "t_pred": 600,
    }

    starting_point = np.array([0, -0.01, 9])

    x_train, x_pred_list = st.data_simulation_new(**simulation_args, sim_data_return=False,
                                                  t_settings_as_real_time=False, starting_point=starting_point)

    parameters = {
        "r_dim": 300,
        "r_to_r_gen_opt": "output_bias",
        "act_fct_opt": "tanh",
        "node_bias_opt": "constant_bias",
        "bias_scale": 0.1,
        "reg_param": 1e-8,
        "w_in_opt": "ordered_sparse",
        "w_in_scale": 1.0,
        "n_rad": 0.1,
        "n_avg_deg": 5.0,
    }

    esn = ESN.ESN_normal()
    x_dim = 3
    esn.build(x_dim, **parameters)

    esn.train(x_train, sync_steps=simulation_args["t_train_sync"], save_res_inp=True, save_r_internal=True, save_r=True,
              save_r_gen=True, save_out=True, save_y_train=True)

    return esn
