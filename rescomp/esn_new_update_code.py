# -*- coding: utf-8 -*-
""" Implements the Echo State Network (ESN) used in Reservoir Computing """

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg.eigen.arpack.arpack \
    import ArpackNoConvergence as _ArpackNoConvergence
import networkx as nx
import pickle
from copy import deepcopy
import gc
import pandas.io.pickle
from . import utilities
from . import simulations
from ._version import __version__


class _ResCompCore(utilities._ESNLogging):
    """
    TBD
    """

    def __init__(self):

        super(_ResCompCore, self).__init__()

        self._r_dim = None
        self._r_gen_dim = None
        self._x_dim = None
        self._y_dim = None

        self._w_out = None

        self._y_to_x_fct = None  # the function to map the output to the input (e.g. when training on difference)
        self._act_fct = None
        self._res_internal_update_fct = None
        self._inp_coupling_fct = None
        self._r_to_r_gen_fct = None
        self._r_gen_to_out_fct = lambda r_gen: self._w_out @ r_gen

        self._leak_factor = None
        self._node_bias = None

        self._last_x = None
        self._last_res_inp = None
        self._last_r_interal = None
        self._last_r = None
        self._last_r_gen = None
        self._last_y = None

        self._saved_res_inp = None
        self._saved_r_internal = None
        self._saved_r = None
        self._saved_r_gen = None
        self._saved_out = None

        self._reg_param = None

        self._default_r = None

    def _res_update(self, x):
        self._last_x = x
        self._last_res_inp = self._inp_coupling_fct(self._last_x)
        self._last_r_interal = self._res_internal_update_fct(self._last_r)

        self._last_r = self._leak_factor * self._last_r + (1 - self._leak_factor) * self._act_fct(self._last_res_inp +
                                                                     self._last_r_interal + self._node_bias)

    def _r_to_r_gen(self):
        self._last_r_gen = self._r_to_r_gen_fct(self._last_r, self._last_x)

    def _res_gen_to_output(self):
        self._last_y = self._r_gen_to_out_fct(self._last_r_gen)

    def _out_to_inp(self):
        return self._y_to_x_fct(self._last_x, self._last_y)

    def drive(self, input, save_res_inp=False, save_r_internal=False, save_r=False, save_r_gen=False, save_out=False):
        steps = input.shape[0]

        if save_res_inp:
            self._saved_res_inp = np.zeros((steps, self._r_dim))
        if save_r_internal:
            self._saved_r_internal = np.zeros((steps, self._r_dim))
        if save_r:
            self._saved_r = np.zeros((steps, self._r_dim))
        if save_r_gen:
            self._saved_r_gen = np.zeros((steps, self._r_gen_dim))
        if save_out:
            self._saved_out = np.zeros((steps, self._y_dim))

        for i_x, x in enumerate(input):
            self._res_update(x)

            if save_res_inp:
                self._saved_res_inp[i_x, :] = self._last_res_inp
            if save_r_internal:
                self._saved_r_internal[i_x, :] = self._last_r_interal
            if save_r:
                self._saved_r[i_x, :] = self._last_r
            if save_r_gen or save_out:
                self._r_to_r_gen()
                if save_r_gen:
                    self._saved_r_gen[i_x, :] = self._last_r_gen
                if save_out:
                    self._res_gen_to_output()
                    self._saved_out[i_x, :] = self._last_y

    def _fit_w_out(self, y_train, r_gen_train):
        self._w_out = np.linalg.solve(
            r_gen_train.T @ r_gen_train + self._reg_param * np.eye(r_gen_train.shape[1]),
            r_gen_train.T @ y_train).T

    def train_synced(self, x_train, y_train, **kwargs):
        kwargs["save_r_gen"] = True
        self.drive(x_train, **kwargs)
        r_gen_train = self._saved_r_gen
        self._fit_w_out(y_train, r_gen_train)

    def loop(self, steps, save_res_inp=False, save_r_internal=False, save_r=False, save_r_gen=False, save_out=False):
        if save_res_inp:
            self._saved_res_inp = np.zeros((steps, self._r_dim))
        if save_r_internal:
            self._saved_r_internal = np.zeros((steps, self._r_dim))
        if save_r:
            self._saved_r = np.zeros((steps, self._r_dim))
        if save_r_gen:
            self._saved_r_gen = np.zeros((steps, self._r_gen_dim))
        if save_out:
            self._saved_out = np.zeros((steps, self._y_dim))

        x_pred = np.zeros((steps, self._x_dim))
        self._r_to_r_gen()
        self._res_gen_to_output()
        x = self._out_to_inp()
        x_pred[0, :] = x

        if save_res_inp:
            self._saved_res_inp[0, :] = self._last_res_inp
        if save_r_internal:
            self._saved_r_internal[0, :] = self._last_r_interal
        if save_r:
            self._saved_r[0, :] = self._last_r
        if save_r_gen:
            self._saved_r_gen[0, :] = self._last_r_gen
        if save_out:
            self._saved_out[0, :] = self._last_y

        for i in range(1, steps):
            self._res_update(x)
            self._r_to_r_gen()
            self._res_gen_to_output()
            x = self._out_to_inp()
            x_pred[i, :] = x
            if save_res_inp:
                self._saved_res_inp[i, :] = self._last_res_inp
            if save_r_internal:
                self._saved_r_internal[i, :] = self._last_r_interal
            if save_r:
                self._saved_r[i, :] = self._last_r
            if save_r_gen:
                self._saved_r_gen[i, :] = self._last_r_gen
            if save_out:
                self._saved_out[i, :] = self._last_y
        return x_pred

    def train(self, x_sync, x_train, y_train, reset_res_state=True, **kwargs):
        if reset_res_state:
            self.reset_r()
        self.drive(x_sync)
        self.train_synced(x_train, y_train, **kwargs)

    # def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
    #     # TODO: doesnt make sense to already hardcode x_train and y_train here
    #     if reset_res_state:
    #         self.reset_r()
    #
    #     if sync_steps > 0:
    #         sync = use_for_train[:sync_steps]
    #         train = use_for_train[sync_steps:]
    #         self.drive(sync)
    #     else:
    #         train = use_for_train
    #
    #     x_train = train[:-1]
    #     y_train = train[1:]
    #
    #     self.train_synced(x_train, y_train, **kwargs)

    def predict(self, use_for_pred, sync_steps=0, reset_res_state=True, **kwargs):
        if reset_res_state:
            self.reset_r()

        if sync_steps > 0:
            sync = use_for_pred[:sync_steps]
            true_data = use_for_pred[sync_steps:]
            self.drive(sync)
        else:
            true_data = use_for_pred

        steps = true_data.shape[0]
        return self.loop(steps, **kwargs), true_data

    def set_r(self, r):
        self._last_r = r

    def reset_r(self):
        self.set_r(self._default_r)

    def get_act_fct_inp(self):
        return self._saved_res_inp + self._saved_r_internal + self._node_bias

    def get_res_inp(self):
        return self._saved_res_inp

    def get_r_internal(self):
        return self._saved_r_internal

    def get_r(self):
        return self._saved_r

    def get_r_gen(self):
        return self._saved_r_gen

    def get_out(self):
        return self._saved_out


class _add_basic_defaults():
    """
    add activation function options, node bias options, leak factor, default_res_state,
    reg_param

    SETS:
        - self._act_fct: r -> r
        - self._node_bias: np.ndarray (shape: r)
        - self._leak_factor: float between 0 and 1
        - self._reg_param: positive float
    """

    def __init__(self):

        self._act_fct_opt = None
        self._act_fct_flag_synonyms = utilities._SynonymDict()
        self._act_fct_flag_synonyms.add_synonyms(0, ["tanh", "tanh_simple", "simple"])
        self._act_fct_flag_synonyms.add_synonyms(1, ["sigmoid"])

        self._node_bias_opt = None
        self._node_bias_flag_synonyms = utilities._SynonymDict()
        self._node_bias_flag_synonyms.add_synonyms(0, ["no_bias"])
        self._node_bias_flag_synonyms.add_synonyms(1, ["random_bias"])
        self._bias_scale = None

    def set_activation_function(self, act_fct_opt="tanh"):
        if type(act_fct_opt) == str:
            self._act_fct_opt = act_fct_opt
            act_fct_flag = self._act_fct_flag_synonyms.get_flag(act_fct_opt)
            if act_fct_flag == 0:
                self._act_fct = np.tanh
            elif act_fct_flag == 1:
                self._act_fct = utilities.sigmoid
        else:
            self._act_fct_opt = "CUSTOM"
            self._act_fct = act_fct_opt

    def set_node_bias(self, node_bias_opt="no_bias", bias_scale=1.0):
        if type(node_bias_opt) == str:
            self._node_bias_opt = node_bias_opt
            node_bias_flag = self._node_bias_flag_synonyms.get_flag(node_bias_opt)
            if node_bias_flag == 0:
                self._node_bias = 0
            elif node_bias_flag == 1:
                self._bias_scale = bias_scale
                self._node_bias = self._bias_scale * np.random.uniform(low=-1.0, high=1.0, size=self._r_dim)
        else:
            self._node_bias_opt = "CUSTOM"
            self._node_bias = node_bias_opt

    def set_leak_factor(self, leak_factor=0.0):
        self._leak_factor = leak_factor

    def set_default_res_state(self, default_res_state=None):
        if default_res_state is None:
            self._default_r = np.zeros(self._r_dim)
        else:
            self._default_r = default_res_state

    def set_reg_param(self, reg_param=1e-8):
        self._reg_param = reg_param


class _add_network_update_fct():
    """
    add network as internal res update function

    SETS:
        - self._res_internal_update_fct r -> r
    """
    def __init__(self):
        self._network = None
        self._res_internal_update_fct = lambda r: self._network @ r

        self._n_type_opt = None
        self._n_rad = None
        self._n_avg_deg = None
        self._n_edge_prob = None
        self._n_type_flag_synonyms = utilities._SynonymDict()
        self._n_type_flag_synonyms.add_synonyms(0, ["random", "erdos_renyi"])
        self._n_type_flag_synonyms.add_synonyms(1, ["scale_free", "barabasi_albert"])
        self._n_type_flag_synonyms.add_synonyms(2, ["small_world", "watts_strogatz"])
        self._n_type_flag_synonyms.add_synonyms(3, ["random_directed", "erdos_renyi_directed"])

    def create_network(self, n_rad=0.1, n_avg_deg=6.0,
                       n_type_opt="erdos_renyi", network_creation_attempts=10):
        if type(n_type_opt) == str:
            self._n_type_opt = n_type_opt
            self._n_rad = n_rad
            self._n_avg_deg = n_avg_deg
            self._n_edge_prob = self._n_avg_deg / (self._r_dim - 1)
            self._n_type_opt = n_type_opt
            n_type_flag = self._n_type_flag_synonyms.get_flag(n_type_opt)
            for i in range(network_creation_attempts):
                try:
                    self._create_network_connections(n_type_flag)
                    self._vary_network()
                except _ArpackNoConvergence:
                    continue
                break
            else:
                raise Exception("Network creation during ESN init failed %d times"
                                % network_creation_attempts)
        else:
            self._n_type_opt = "CUSTOM"
            self._network = n_type_opt

    def _create_network_connections(self, n_type_flag):
        """ Generate the baseline random network to be scaled

        Specification done via protected members
        """

        if n_type_flag == 0:
            network = nx.fast_gnp_random_graph(self._r_dim, self._n_edge_prob,
                                               seed=np.random)
        elif n_type_flag == 1:
            network = nx.barabasi_albert_graph(self._r_dim,
                                               int(self._n_avg_deg / 2),
                                               seed=np.random)
        elif n_type_flag == 2:
            network = nx.watts_strogatz_graph(self._r_dim,
                                              k=int(self._n_avg_deg), p=0.1,
                                              seed=np.random)
        elif n_type_flag == 3:
            network = nx.fast_gnp_random_graph(self._r_dim, self._n_edge_prob,
                                               seed=np.random, directed=True)
        else:
            raise Exception("the network type %s is not implemented" %
                            str(self._n_type_opt))

        self._network = nx.to_numpy_array(network)

    def _vary_network(self, network_variation_attempts=10):
        """ Varies the weights of self._network, while conserving the topology.

        The non-zero elements of the adjacency matrix are uniformly randomized,
        and the matrix is scaled (self.scale_network()) to self.spectral_radius.

        Specification done via protected members

        """

        # contains tuples of non-zero elements:
        arg_binary_network = np.argwhere(self._network)

        for i in range(network_variation_attempts):
            try:
                # uniform entries from [-0.5, 0.5) at non-zero locations:
                rand_shape = self._network[self._network != 0.].shape
                self._network[
                    arg_binary_network[:, 0], arg_binary_network[:, 1]] = \
                    np.random.random(size=rand_shape) - 0.5

                self._scale_network()

            except _ArpackNoConvergence:
                self.logger.error(
                    'Network Variaion failed! -> Try agin!')
                continue
            break
        else:
            # TODO: Better logging of exceptions
            self.logger.error("Network variation failed %d times"
                              % network_variation_attempts)
            raise _ArpackNoConvergence

    def _scale_network(self):
        """ Scale self._network, according to desired spectral radius.

        Can cause problems due to non converging of the eigenvalue evaluation

        Specification done via protected members

        """
        self._network = scipy.sparse.csr_matrix(self._network)
        try:
            eigenvals = scipy.sparse.linalg.eigs(
                self._network, k=1, v0=np.ones(self._r_dim),
                maxiter=1e3 * self._r_dim)[0]
        except _ArpackNoConvergence:
            self.logger.error('Eigenvalue calculation in scale_network failed!')
            raise

        maximum = np.absolute(eigenvals).max()
        self._network = ((self._n_rad / maximum) * self._network)


class _add_dyn_sys_update_fct():
    """
    Use a Dynamical System as res_internal_update_fct
    SETS:
        - self._res_internal_update_fct r -> r
    """
    def __init__(self):
        self._dyn_sys_opt = None
        self._dyn_sys_dt = None
        self._dyn_sys_scale_factor = None
        self._dyn_sys_other_params = None
        self._dyn_sys_flag_synonyms = utilities._SynonymDict()
        self._dyn_sys_flag_synonyms.add_synonyms(0, ["L96"])
        self._dyn_sys_flag_synonyms.add_synonyms(1, ["KS"])

    def create_dyn_sys_upd_fct(self, dyn_sys_opt="L96", dyn_sys_dt=0.1, scale_factor=1.0, dyn_sys_other_params=(5., )):
        if type(dyn_sys_opt) == str:
            self._dyn_sys_opt = dyn_sys_opt
            self._dyn_sys_dt = dyn_sys_dt
            self._dyn_sys_scale_factor = scale_factor
            self._dyn_sys_other_params = dyn_sys_other_params
            dyn_sys_flag = self._dyn_sys_flag_synonyms.get_flag(dyn_sys_opt)
            if dyn_sys_flag == 0:
                L96_force = dyn_sys_other_params[0]

                def _lorenz_96(x):
                    return simulations._lorenz_96(x, force=L96_force)

                def f_L96(x):
                    return simulations._runge_kutta(_lorenz_96, dyn_sys_dt, x)

                self._res_internal_update_fct = lambda r: (f_L96(r)-r) * self._dyn_sys_scale_factor

            elif dyn_sys_flag == 1:
                KS_system_size = dyn_sys_other_params[0]

                def f_KS(x):
                    return simulations._kuramoto_sivashinsky(self._r_dim, system_size=KS_system_size, dt=dyn_sys_dt,
                                                             time_steps=2, starting_point=x)[-1]

                self._res_internal_update_fct = lambda r: (f_KS(r) - r) * self._dyn_sys_scale_factor

            # self._res_internal_update_fct = lambda r: self._dyn_sys_scale_factor * self._res_internal_update_fct(r)
        else:
            self._dyn_sys_opt = "CUSTOM"
            self._res_internal_update_fct = dyn_sys_opt


class _add_basic_r_to_rgen():
    """
    add basic r to r_gen behaviour
    SETS:
    - self._r_to_r_gen_fct: r -> r_gen (any dimension)
    """
    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                self._r_to_r_gen_fct = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T
                self._r_to_r_gen_fct = temp
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            self._r_to_r_gen_fct = r_to_r_gen_opt

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), None).shape[0]


class _add_model_r_to_rgen():
    """
    add r to r_gen behaviour with hybrid-output model
    SETS:
    - self._r_to_r_gen_fct: r -> r_gen (any dimension)
    """
    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])

        self.model = None

    def set_model(self, model):
        self.model = model

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                _r_to_r_gen_fct_no_model = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T
                _r_to_r_gen_fct_no_model = temp
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            _r_to_r_gen_fct_no_model = r_to_r_gen_opt

        self._r_to_r_gen_fct = lambda r, x: np.hstack((_r_to_r_gen_fct_no_model(r, x), self.model(x)))

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[0]


class _add_w_in():
    """
    add basic w_in behavior

    Only creates a self._w_in. What you do with it is still open
    """
    def __init__(self):

        self._w_in_opt = None
        self._w_in_scale = None
        self._w_in_flag_synonyms = utilities._SynonymDict()
        self._w_in_flag_synonyms.add_synonyms(0, ["random_sparse"])
        self._w_in_flag_synonyms.add_synonyms(1, ["ordered_sparse"])
        self._w_in_flag_synonyms.add_synonyms(2, ["random_dense_uniform"])
        self._w_in_flag_synonyms.add_synonyms(3, ["random_dense_gaussian"])

    def create_w_in(self, w_in_opt, w_in_scale=1.0):
        self.logger.debug("Create w_in")

        if type(w_in_opt) == str:
            self._w_in_scale = w_in_scale
            self._w_in_opt = w_in_opt
            w_in_flag = self._w_in_flag_synonyms.get_flag(w_in_opt)

            if w_in_flag == 0:
                self._w_in = np.zeros((self._r_dim, self._x_dim))
                for i in range(self._r_dim):
                    random_x_coord = np.random.choice(np.arange(self._x_dim))
                    self._w_in[i, random_x_coord] = np.random.uniform(
                        low=-self._w_in_scale,
                        high=self._w_in_scale)

            elif w_in_flag == 1:
                self._w_in = np.zeros((self._r_dim, self._x_dim))
                dim_wise = np.array([int(self._r_dim / self._x_dim)] * self._x_dim)
                dim_wise[:self._r_dim % self._x_dim] += 1
                s = 0
                dim_wise_2 = dim_wise[:]
                for i in range(len(dim_wise_2)):
                    s += dim_wise_2[i]
                    dim_wise[i] = s
                dim_wise = np.append(dim_wise, 0)
                for d in range(self._x_dim):
                    for i in range(dim_wise[d - 1], dim_wise[d]):
                        self._w_in[i, d] = np.random.uniform(
                            low=-self._w_in_scale,
                            high=self._w_in_scale)

            elif w_in_flag == 2:
                self._w_in = np.random.uniform(low=-self._w_in_scale,
                                           high=self._w_in_scale,
                                           size=(self._r_dim, self._x_dim))

            elif w_in_flag == 3:
                self._w_in = self._w_in_scale*np.random.randn(self._r_dim, self._x_dim)

        else:
            self._w_in_opt = "CUSTOM"
            self._w_in = w_in_opt


class _add_standard_input_coupling():
    """
    add normal input coupling via
    SETS:
        - self._inp_coupling_fct
    """
    def __init__(self):
        self._inp_coupling_fct = lambda x: self._w_in @ x


class _add_standard_y_to_x():
    """
    Function from output to input for loop is identity
    """
    def __init__(self):
        self._y_to_x_fct = lambda x, y: y


class _add_y_diff_to_x():
    """
    Function from output to input where output is difference
    """
    def __init__(self):
        self._dt_difference = None
        self._y_to_x_fct = lambda x, y: x + y * self._dt_difference

    def set_dt_difference(self, dt_difference=0.1):
        self._dt_difference = dt_difference


class ESN_normal(_ResCompCore, _add_basic_defaults, _add_network_update_fct, _add_basic_r_to_rgen,
                 _add_w_in, _add_standard_input_coupling, _add_standard_y_to_x):
    """
    Pretty standard ESN class
    """
    def __init__(self):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_network_update_fct.__init__(self)
        _add_basic_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]
        super(ESN_normal, self).train(sync, x_train, y_train, reset_res_state=reset_res_state, **kwargs)

    def build(self, x_dim, r_dim=500, n_rad=0.1, n_avg_deg=6.0, n_type_opt="erdos_renyi", network_creation_attempts=10,
              r_to_r_gen_opt="linear", act_fct_opt="tanh", node_bias_opt="no_bias", bias_scale=1.0, leak_factor=0.0,
              w_in_opt="random_sparse", w_in_scale=1.0, default_res_state=None, reg_param=1e-8, network_seed=None,
              bias_seed=None, w_in_seed=None):

        self.logger.debug("Building ESN Archtecture")

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        if network_seed is not None:
            with utilities.temp_seed(network_seed):
                self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                    network_creation_attempts=network_creation_attempts)
        else:
            self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                network_creation_attempts=network_creation_attempts)
        self.set_r_to_r_gen_fct(r_to_r_gen_opt=r_to_r_gen_opt)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)


class ESN_dynsys(_ResCompCore, _add_basic_defaults, _add_dyn_sys_update_fct, _add_basic_r_to_rgen,
                 _add_w_in, _add_standard_input_coupling, _add_standard_y_to_x):
    """
    Use a Dynamical System as res_internal_update_fct
    """
    def __init__(self,):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_dyn_sys_update_fct.__init__(self)
        _add_basic_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]
        super(ESN_dynsys, self).train(sync, x_train, y_train, reset_res_state=reset_res_state, **kwargs)

    def build(self, x_dim, r_dim=500, dyn_sys_opt="L96", dyn_sys_dt=0.1, scale_factor=1., dyn_sys_other_params=(5.,),
              r_to_r_gen_opt="linear", act_fct_opt="tanh", node_bias_opt="no_bias", bias_scale=1.0, leak_factor=0.0, w_in_opt="random_sparse",
              w_in_scale=1.0, default_res_state=None, reg_param=1e-8, bias_seed=None, w_in_seed=None):

        self.logger.debug("Building ESN Archtecture")

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        self.create_dyn_sys_upd_fct(dyn_sys_opt=dyn_sys_opt, dyn_sys_dt=dyn_sys_dt, scale_factor=scale_factor,
                                    dyn_sys_other_params=dyn_sys_other_params)

        self.set_r_to_r_gen_fct(r_to_r_gen_opt=r_to_r_gen_opt)
        self.set_activation_function(act_fct_opt=act_fct_opt)
        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)


class ESN_difference(_ResCompCore, _add_basic_defaults, _add_network_update_fct, _add_basic_r_to_rgen,
                 _add_w_in, _add_standard_input_coupling, _add_y_diff_to_x):
    """
    Use a Dynamical System as res_internal_update_fct
    """

    def __init__(self, ):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_network_update_fct.__init__(self)
        _add_basic_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_y_diff_to_x.__init__(self)

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = (train[1:] - x_train)/self._dt_difference

        super(ESN_difference, self).train(sync, x_train, y_train, reset_res_state=reset_res_state, **kwargs)

    def build(self, x_dim, r_dim=500, n_rad=0.1, n_avg_deg=6.0, n_type_opt="erdos_renyi", network_creation_attempts=10,
              r_to_r_gen_opt="linear", act_fct_opt="tanh", node_bias_opt="no_bias", bias_scale=1.0, leak_factor=0.0,
              w_in_opt="random_sparse", w_in_scale=1.0, default_res_state=None, reg_param=1e-8, network_seed=None,
              bias_seed=None, w_in_seed=None, dt_difference=0.1):

        self.logger.debug("Building ESN Archtecture")

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        if network_seed is not None:
            with utilities.temp_seed(network_seed):
                self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                    network_creation_attempts=network_creation_attempts)
        else:
            self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                network_creation_attempts=network_creation_attempts)
        self.set_r_to_r_gen_fct(r_to_r_gen_opt=r_to_r_gen_opt)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)

        self.set_dt_difference(dt_difference=dt_difference)


class ESN_output_hybrid(_ResCompCore, _add_basic_defaults, _add_network_update_fct, _add_model_r_to_rgen,
                 _add_w_in, _add_standard_input_coupling, _add_standard_y_to_x):
    """
    ESN with output hybrid model
    """
    def __init__(self):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_network_update_fct.__init__(self)
        _add_model_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]
        super(ESN_output_hybrid, self).train(sync, x_train, y_train, reset_res_state=reset_res_state, **kwargs)

    def set_w_out_to_only_model(self):
        '''
        Just a experimental function that wires all connection in w_out to only read
        the model based prediction (if output_hybrid)
        :return:
        '''
        # modify _w_out:
        n_dim_mod = self._r_gen_dim
        x_dim = self._x_dim
        matrix = np.zeros((x_dim, n_dim_mod))
        for i in range(0, x_dim):
            to_add = np.zeros(x_dim)
            to_add[i] = 1
            matrix[:, n_dim_mod - x_dim + i] = to_add
        self._w_out = matrix

    def build(self, x_dim, r_dim=500, n_rad=0.1, n_avg_deg=6.0, n_type_opt="erdos_renyi", network_creation_attempts=10,
              r_to_r_gen_opt="linear", model=lambda x: x, act_fct_opt="tanh", node_bias_opt="no_bias", bias_scale=1.0, leak_factor=0.0,
              w_in_opt="random_sparse", w_in_scale=1.0, default_res_state=None, reg_param=1e-8, network_seed=None,
              bias_seed=None, w_in_seed=None):

        self.logger.debug("Building ESN Archtecture")

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        if network_seed is not None:
            with utilities.temp_seed(network_seed):
                self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                    network_creation_attempts=network_creation_attempts)
        else:
            self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                network_creation_attempts=network_creation_attempts)

        self.set_model(model)
        self.set_r_to_r_gen_fct(r_to_r_gen_opt=r_to_r_gen_opt)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)
