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
from ._version import __version__


# dictionary defining synonyms for the different methods to generalize the
# reservoir state r(t) to a nonlinear fit for _w_out

class _ESNCore(utilities._ESNLogging):
    """ The non-reducible core of ESN RC training and prediction

    While technically possible to be used on it's own, this is very much not
    recommended. Use the child class(es) instead.

    """

    def __init__(self):

        super().__init__()

        self._w_in = None

        self._network = None

        self._w_out = None

        self._act_fct = None

        self._w_out_fit_flag_synonyms = utilities._SynonymDict()
        self._w_out_fit_flag_synonyms.add_synonyms(0, ["linear_r", "simple"])
        self._w_out_fit_flag_synonyms.add_synonyms(1, "linear_and_square_r")
        self._w_out_fit_flag_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._w_out_fit_flag_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._w_out_fit_flag_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])

        self._w_out_fit_flag = None
        self._w_out_fit_flag_str = None

        self._last_r = None
        self._last_r_gen = None

        self._loc_nbhd = None

        self._reg_param = None

        self.x_train_pred = None

    def set_w_out_fit_flag(self, w_out_fit_flag="simple"):
        """
        Simple func to set the w_out_fit_flag-> will be reseted if _train_synced is called
        """
        self._w_out_fit_flag = \
            self._w_out_fit_flag_synonyms.get_flag(w_out_fit_flag)
        self._w_out_fit_flag_str = w_out_fit_flag
        self.logger.debug(f"_w_out_fit_flag set to {w_out_fit_flag}")

    def reset_res_state(self, res_state=None):
        """
        (re)set the reservoir state
        Args:
            res_state (np.ndarray): The state of the reservoir nodes , shape (N_dim,)
        """
        if res_state is None:
            self._last_r = np.zeros(self._network.shape[0])
        else:
            if len(res_state.shape) != 1:
                raise Exception("res_state has to be 1-D")
            elif res_state.size != self._network.shape[0]:
                raise Exception("res_state size does not fit network size")
            else:
                self._last_r = res_state

    def synchronize(self, x, save_r=False):
        """ Synchronize the reservoir state with the input time series

        This is usually done automatically in the training and prediction
        functions.

        Args:
            x (np.ndarray): Input data to be used for the synchronization,
                shape (T, d)
            save_r (bool): If true, saves and returns r

        Returns:
            np.ndarray_or_None: All r states if save_r is True, None if False

        """
        self.logger.debug('Start syncing the reservoir state')

        if self._last_r is None:
            # self._last_r = np.zeros(self._network.shape[0])
            self.reset_res_state()

        if save_r:
            r = np.zeros((x.shape[0], self._network.shape[0]))
            r[0] = self._act_fct(x[0], self._last_r)
            for t in np.arange(x.shape[0] - 1):
                r[t + 1] = self._act_fct(x[t + 1], r[t])
            self._last_r = deepcopy(r[-1])
            return r
        else:
            for t in np.arange(x.shape[0]):
                self._last_r = self._act_fct(x[t], self._last_r)
            return None

    def _r_to_generalized_r(self, r):
        """ Convert the internal reservoir state r into the generalized r_gen

        r_gen is the (nonlinear) transformation applied to r before the output
        is calculated.
        The type of transformation is set via the self._w_out_fit_flag parameter

        Args:
            r (np.ndarray): the r to convert to r_gen. Both the 2dim r of shape
                (T, d) as well as the 1dim _last_r of shape (d,) should be
                supported

        Returns:
            np.ndarray: r_gen, the transformed r

        """
        # This needs to work for
        if self._w_out_fit_flag == 0:
            return r
        elif self._w_out_fit_flag == 1:
            return np.hstack((r, r ** 2))
        elif self._w_out_fit_flag == 2:
            if len(r.shape) == 1:
                return np.hstack((r, 1))
            elif len(r.shape) == 2:
                return np.hstack((r, np.ones(r.shape[0])[:, None]))
        elif self._w_out_fit_flag == 3:
            if len(r.shape) == 1:
                return np.hstack((np.hstack((r, r ** 2)), 1))
            elif len(r.shape) == 2:
                return np.hstack((np.hstack((r, r ** 2)),
                                  np.ones(r.shape[0])[:, None]))
        elif self._w_out_fit_flag == 4:
            r_gen = np.copy(r).T
            r_gen[::2] = r.T[::2] ** 2
            return r_gen.T
        else:
            raise Exception("self._w_out_fit_flag incorrectly specified")

    def _fit_w_out(self, x_train, r, save_x_train_pred=False):
        """ Fit the output matrix self._w_out after training

        Uses linear regression and Tikhonov regularization.

        Args:
            x_train (np.ndarray): get y_train via x_train[1:], y_train = Desired prediction from the reservoir states
            r (np.ndarray): reservoir states
        Returns:
            np.ndarray: r_gen, generalized nonlinear transformed r

        """
        # Note: in an older version y_train was obtained from x_train in _train_synced, and then
        # parsed directly to _fit_w_out.
        # This is changed in order to allow for an easy override of _fit_w_out in variations of
        # the ESN class (e.g. ESNHybrid), that include other non-rc predictions (e.g. x_new = model(x_old))
        # into an "extended r_gen" = concat(r_gen, model(x)), that is finally fitted to y_train.

        # Note: This is slightly different than the old ESN as y_train was as
        # long as x_train, but shifted by one time step. Hence to get the same
        # results as for the old ESN one has to specify an x_train one time step
        # longer than before. Nonetheless, it's still true that r[t] is
        # calculated from x[t] and used to calculate y[t] (all the same t)

        y_train = x_train[1:]
        self.logger.debug('Fit _w_out according to method %s' %
                          str(self._w_out_fit_flag))

        r_gen = self._r_to_generalized_r(r)

        # If we are using local states we only want to use the core dimensions
        # for the fit of W_out, i.e. the dimensions where the corresponding
        # locality matrix == 2
        if self._loc_nbhd is None:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train).T
        else:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train[:, self._loc_nbhd == 2]).T

        if save_x_train_pred:
            print(r_gen.shape)
            print(self._w_out.shape)
            self.x_train_pred = (self._w_out @ r_gen.T).T

        return r_gen

    def _train_synced(self, x_train, w_out_fit_flag="simple", save_x_train_pred=False):
        """ Train a synchronized reservoir

        Args:
            x_train (np.ndarray): input to be used for the training, shape (T,d)
            w_out_fit_flag (str): Type of nonlinear transformation applied to
                the reservoir states r to be used during the fit (and future
                prediction

        Returns:
            tuple: 2-element tuple containing:

            - **r** (*np.ndarray*) reservoir states
            - **r_gen** (*np.ndarray*): generalized reservoir states

        """

        # self._w_out_fit_flag = \
        #     self._w_out_fit_flag_synonyms.get_flag(w_out_fit_flag)
        if self._w_out_fit_flag is None:
            self.set_w_out_fit_flag(w_out_fit_flag)

        self.logger.debug('Start training')

        # The last value of r can't be used for the training, see comment below
        r = self.synchronize(x_train[:-1], save_r=True)

        r_gen = self._fit_w_out(x_train, r, save_x_train_pred=save_x_train_pred)

        return r, r_gen

    def _predict_step(self, x=None):
        """ Predict a single time step

        Assumes a synchronized reservoir.
        Changes self._last_r and self._last_r_gen to stay synchronized to the
        new system state y

        Args:
            x (np.ndarray): input for the d-dim. system, shape (d,). If x is
            None the internal reservoir states will be used to generate x
            using w_out

        Returns:
            np.ndarray: y, the next time step as predicted from last_x, _w_out
            and _last_r, shape (d,)
        
        """

        if x is None:
            x = self._w_out @ self._r_to_generalized_r(self._last_r)

        self._last_r = self._act_fct(x, self._last_r)
        self._last_r_gen = self._r_to_generalized_r(self._last_r)

        y = self._w_out @ self._last_r_gen

        if self._loc_nbhd is not None:
            temp = np.empty(self._loc_nbhd.shape)
            temp[:] = np.nan
            temp[self._loc_nbhd == 2] = y
            y = temp

        return y


class ESN(_ESNCore):
    """ Implements basic RC functionality

    This is to be written such that one can easily implement both "normal" RC as
    well as local RC, or other generalizations, by using this class as a
    building block to be called with the right training and prediction data

    """

    def __init__(self):

        super().__init__()

        # create_network() assigns values to:
        self._n_dim = None  # network_dimension
        self._n_rad = None  # network_spectral_radius
        self._n_avg_deg = None  # network_average_degree
        self._n_edge_prob = None
        self._n_type_flag = None  # network_type
        self._n_type_flag_str = None
        self._network = self._network

        # _create_w_in() which is called from train() assigns values to:
        self._w_in_sparse = None
        self._w_in_ordered = None
        self._w_in_scale = None
        self._w_in = self._w_in

        # set_activation_function assigns values to:
        self._bias_scale = None
        self._bias = None
        self._act_fct_flag = None
        self._act_fct_flag_str = None
        self._act_fct = self._act_fct
        self._normal_tanh_nodes = None
        self._squared_tanh_nodes = None
        self._leak_fct = None

        # train() assigns values to:
        self._x_dim = None  # Typically called d
        self._reg_param = self._reg_param
        self._w_out = self._w_out
        # if save_input is true, train() also assigns values to:
        self._x_train_sync = None  # data used to sync before training
        self._x_train = None  # data used for training to fit w_out
        # if save_r is true, train() also assigns values to:
        self._r_train = None
        self._r_train_gen = None

        # predict() assigns values to:
        self._y_pred = None
        # if save_input is true, predict() also assigns values to:
        self._x_pred_sync = None  # data used to sync before prediction
        self._y_test = None  # data used to compare the prediction to
        # if save_r is true, predict() also assigns values to:
        self._r_pred = None
        self._r_pred_gen = None

        # Dictionary defining synonyms for the different ways to choose the
        # activation function. Internally the corresponding integers are used.
        # If you add/change a flag here, please also change it in the
        # train docstring
        self._act_fct_flag_synonyms = utilities._SynonymDict()
        self._act_fct_flag_synonyms.add_synonyms(0, ["tanh", "tanh_simple", "simple"])
        self._act_fct_flag_synonyms.add_synonyms(1, "tanh_bias")
        self._act_fct_flag_synonyms.add_synonyms(2, "tanh_squared")
        self._act_fct_flag_synonyms.add_synonyms(3, ["mixed", "mix"])
        self._act_fct_flag_synonyms.add_synonyms(4, ["sigmoid"])

        # Dictionary defining synonyms for the different ways to create the
        # network. Internally the corresponding integers are used
        # If you add/change a flag here, please also change it in the
        # create_network docstring
        self._n_type_flag_synonyms = utilities._SynonymDict()
        self._n_type_flag_synonyms.add_synonyms(0, ["random", "erdos_renyi"])
        self._n_type_flag_synonyms.add_synonyms(1, ["scale_free", "barabasi_albert"])
        self._n_type_flag_synonyms.add_synonyms(2, ["small_world", "watts_strogatz"])
        self._n_type_flag_synonyms.add_synonyms(3, ["random_directed", "erdos_renyi_directed"])

        # Set during class creation, used during loading from pickle
        self._rescomp_version = __version__

        self.logger.debug("Create ESN instance")

    def _create_w_in(self):
        """  Create the input matrix w_in

        Specification done via protected members

        """
        self.logger.debug("Create w_in")

        if self._w_in_sparse and not self._w_in_ordered:
            self._w_in = np.zeros((self._n_dim, self._x_dim))
            for i in range(self._n_dim):
                random_x_coord = np.random.choice(np.arange(self._x_dim))
                self._w_in[i, random_x_coord] = np.random.uniform(
                    low=-self._w_in_scale,
                    high=self._w_in_scale)  # maps input values to reservoir

        elif self._w_in_sparse and self._w_in_ordered:
            self._w_in = np.zeros((self._n_dim, self._x_dim))
            dim_wise = np.array([int(self._n_dim / self._x_dim)] * self._x_dim)
            dim_wise[:self._n_dim % self._x_dim] += 1

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
                        high=self._w_in_scale)  # maps input values to reservoir
        else:
            self._w_in = np.random.uniform(low=-self._w_in_scale,
                                           high=self._w_in_scale,
                                           size=(self._n_dim, self._x_dim))

    def _create_w_in_alternative(self, x_dim, type="rect", normalize=True, w_in_scale=1.0):
        self._x_dim = x_dim
        self._w_in = np.zeros((self._n_dim, self._x_dim))

        ratio = self._n_dim/self._x_dim

        if type == "rect":
            for i_x in range(x_dim):
                i_n_min = int(i_x * ratio)
                i_n_max = int((i_x + 1) * ratio)
                self._w_in[i_n_min : i_n_max, i_x] = 1

        elif type == "gauss":
            def gaussian(x, x0, sigma):
                return np.exp(-np.power((x - x0) / sigma, 2.) / 2.)

            for i_x in range(x_dim):
                i_n_min = i_x * ratio
                i_n_max = (i_x + 1) * ratio
                i_n_mid = int((i_n_max + i_n_min)/2)
                for i_n in range(self._n_dim):
                    self._w_in[i_n, i_x] = gaussian(i_n, i_n_mid, sigma=ratio/2)

        elif type == "alternating":
            i_x = 0
            for i_n in range(self._n_dim):
                self._w_in[i_n, i_x] = 1
                i_x = (i_x+1) % self._x_dim

        elif type == "sinus":
            for i_x in range(x_dim):
                period = (2*np.pi)/self._n_dim
                start = self._n_dim/(x_dim*2) * i_x
                for i_n in range(self._n_dim):
                    self._w_in[i_n, i_x] = np.sin(period*(i_n - start))

        # normalize
        if normalize:
            norm = np.linalg.norm(self._w_in, ord=2)  # largest singular value
            self._w_in = self._w_in/norm

        if self._w_in_scale is None:
            self._w_in_scale = w_in_scale

        self._w_in *= self._w_in_scale

    def set_w_out(self, w_out=None):
        """
        Func to set a w_out to be able to run the loop (loop the reservoir dynamics without training data)
        """
        self.logger.debug("Setting self._w_out with the set_w_out function")

        r_aux = np.zeros(self._n_dim)
        res_gen_size = self._r_to_generalized_r(r_aux).size

        if w_out is None:
            self._w_out = np.ones((self._x_dim, res_gen_size))  # set every entry of _w_out to one (just so that the program runs)
            self.logger.debug("Setting all self._w_out entries to 1")
        else:
            if w_out.shape != (self._x_dim, res_gen_size):
                raise Exception("Shape of parsed w_out does not fit the requirements")
            else:
                self._w_out = w_out

    def create_network(self, n_dim=500, n_rad=0.1, n_avg_deg=6.0,
                       n_type_flag="erdos_renyi", network_creation_attempts=10):
        """ Creates the internal network used as reservoir in RC

        Args:
            n_dim (int): Nr. of nodes in the network
            n_rad (float): Spectral radius of the network. Must be >0. Should be
                < 1 to satisfy the echo state property
            n_avg_deg (float): Average node degree (number of connections). Used
                in the random network generations
            n_type_flag (int_or_str): Type of Network to be generated. Possible
                flags and their synonyms are:

                - "random", "erdos_renyi"
                - "scale_free", "barabasi_albert"
                - "small_world", "watts_strogatz"
            network_creation_attempts (int): How often the network generation
                should be attempted. It can fail due to a not converging
                eigenvalue calculation

        """

        self.logger.debug("Create network")

        self._n_dim = n_dim
        self._n_rad = n_rad
        self._n_avg_deg = n_avg_deg
        self._n_edge_prob = self._n_avg_deg / (self._n_dim - 1)
        self._n_type_flag = self._n_type_flag_synonyms.get_flag(n_type_flag)
        self._n_type_flag_str = n_type_flag

        for i in range(network_creation_attempts):
            try:
                self._create_network_connections()
                self._vary_network()
            except _ArpackNoConvergence:
                continue
            break
        else:
            raise Exception("Network creation during ESN init failed %d times"
                            % network_creation_attempts)

    def _create_network_connections(self):
        """ Generate the baseline random network to be scaled

        Specification done via protected members

        """

        if self._n_type_flag == 0:
            network = nx.fast_gnp_random_graph(self._n_dim, self._n_edge_prob,
                                               seed=np.random)
        elif self._n_type_flag == 1:
            network = nx.barabasi_albert_graph(self._n_dim,
                                               int(self._n_avg_deg / 2),
                                               seed=np.random)
        elif self._n_type_flag == 2:
            network = nx.watts_strogatz_graph(self._n_dim,
                                              k=int(self._n_avg_deg), p=0.1,
                                              seed=np.random)
        elif self._n_type_flag == 3:
            network = nx.fast_gnp_random_graph(self._n_dim, self._n_edge_prob,
                                               seed=np.random, directed = True)
        else:
            raise Exception("the network type %s is not implemented" %
                            str(self._n_type_flag))

        self._network = nx.to_numpy_array(network)
        # meh = nx.to_numpy_matrix(network)
        # self._network = np.asarray(nx.to_numpy_matrix(network))

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
                self._network, k=1, v0=np.ones(self._n_dim),
                maxiter=1e3 * self._n_dim)[0]
        except _ArpackNoConvergence:
            self.logger.error('Eigenvalue calculation in scale_network failed!')
            raise

        maximum = np.absolute(eigenvals).max()
        self._network = ((self._n_rad / maximum) * self._network)

    # def set_network(self, network):
    #     """ Set the network by passing a matrix or a file path
    #
    #     Calculates the corresponding properties to match the new network
    #
    #     Args:
    #         network (nd_array_or_csr_matrix_str):
    #
    #     """
    #     raise Exception("Not yet implemented")

    def set_activation_function(self, act_fct_flag, bias_scale=0, mix_ratio=0.5, leak_fct=0.0, custom_act_fct=None):
        """ Set the activation function corresponding to the act_fct_flag

        Args:
            act_fct_flag (int_or_str): flag corresponding to the activation
                function one wants to use, see :func:`~esn.ESN.train` for a
                list of possible flags
            bias_scale (float): Bias to be used in some activation functions
                (currently only used in :func:`~esn.ESN._act_fct_tanh_bias`)
            leak_fct (float) a leak factor between 0 and 1: Determines the memory of the reservoir dynamics

        """
        self.logger.debug("Set activation function to flag: %s" % act_fct_flag)

        # self._act_fct_flag = act_fct_flag
        self._act_fct_flag = self._act_fct_flag_synonyms.get_flag(act_fct_flag)
        self._act_fct_flag_str = act_fct_flag

        self._bias_scale = bias_scale
        self._bias = self._bias_scale * np.random.uniform(low=-1.0, high=1.0,
                                                          size=self._n_dim)
        self._leak_fct = leak_fct

        if self._act_fct_flag == 0:
            # self._act_fct = self._act_fct_tanh_simple
            _act_fct = self._act_fct_tanh_simple
        elif self._act_fct_flag == 1:
            # self._act_fct = self._act_fct_tanh_bias
            _act_fct = self._act_fct_tanh_bias
        elif self._act_fct_flag == 2:
            # self._act_fct = self._act_fct_tanh_squared
            _act_fct = self._act_fct_tanh_squared
        elif self._act_fct_flag == 3:
            self.setup_mix(mix_ratio)
            # self._act_fct = self._act_fct_mixed
            _act_fct = self._act_fct_mixed
        elif self._act_fct_flag == 4:
            # self._act_fct = self._act_fct_sigmoid
            _act_fct = self._act_fct_sigmoid

        else:
            raise Exception('self._act_fct_flag %s does not have a activation '
                            'function implemented!' % str(self._act_fct_flag))
        if custom_act_fct is None:
            self._act_fct = lambda x, r: self._leak_fct*r + (1-self._leak_fct)*_act_fct(x, r)
        else:
            self._act_fct = lambda x, r:  custom_act_fct(self, x, r)

    def _act_fct_tanh_simple(self, x, r):
        """ Standard activation function of the elementwise np.tanh()

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray: n-dim

        """

        return np.tanh(self._w_in @ x + self._network @ r)

    def _act_fct_tanh_bias(self, x, r):
        """ Activation function of the elementwise np.tanh() with added bias

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray n-dim

        """

        return np.tanh(self._w_in @ x + self._network @ r + self._bias)

    def _act_fct_tanh_squared(self, x, r):
        """ Activation function of the elementwise np.tanh() squared with added
            bias.

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray n-dim

        """

        return np.tanh(self._w_in @ x + self._network @ r + self._bias) ** 2

    def _act_fct_mixed(self, x, r):
        """ Different activation functions for different nodes

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray n-dim

        """
        new_r = np.zeros(self._n_dim)

        new_r[self._normal_tanh_nodes] = self._act_fct_tanh_bias(x, r)[
            self._normal_tanh_nodes]

        new_r[self._squared_tanh_nodes] = self._act_fct_tanh_squared(x, r)[
            self._squared_tanh_nodes]

        return new_r

    def _act_fct_sigmoid(self, x, r):
        """ Standard activation function of the elementwise np.tanh()

        Args:
            x (np.ndarray): d-dim input
            r (np.ndarray): n-dim network states

        Returns:
            np.ndarray: n-dim

        """
        a = self._w_in @ x + self._network @ r
        return 1 / (1 + np.exp(-a))  # equals sigmoid(a)

    def setup_mix(self, mix_ratio):
        self._normal_tanh_nodes = []
        self._squared_tanh_nodes = []

        for d in range(self._x_dim):
            dimwise_nodes = np.nonzero(self._w_in[:, d])[0]
            self._normal_tanh_nodes.extend(
                dimwise_nodes[:round(len(dimwise_nodes) * mix_ratio)])
            self._squared_tanh_nodes.extend(
                dimwise_nodes[round(len(dimwise_nodes) * mix_ratio):])

    def create_input_matrix(self, x_dim, w_in_scale=1.0, w_in_sparse=True, w_in_ordered=False):
        ''' create the input matrix _w_in. Can be used to create the _w_in before "train", otherwise it is called in "train"
        Args:
            x_dim (int): dimension of the input data
            w_in_scale (float): maximum absolute value of the (random) w_in
                elements
            w_in_sparse (bool): If true, creates w_in such that one element in
                each row is non-zero (Lu,Hunt, Ott 2018)
            w_in_orderd (bool): If true and w_in_sparse is true, creates w_in
                such that elements are ordered by dimension and number of
                elements for each dimension is equal (as far as possible)
        '''
        self._w_in_scale = w_in_scale
        self._w_in_sparse = w_in_sparse
        self._w_in_ordered = w_in_ordered
        self._x_dim = x_dim
        self._create_w_in()

    def train(self, x_train, sync_steps, reg_param=1e-5, w_in_scale=1.0,
              w_in_sparse=True, w_in_ordered=False, w_in_no_update=False,
              act_fct_flag='tanh_simple', bias_scale=0, mix_ratio=0.5,
              save_r=False, save_input=False, w_out_fit_flag="simple",
              loc_nbhd=None, save_x_train_pred=False):
        """ Synchronize, then train the reservoir

        Args:
            x_train (np.ndarray): Input data used to synchronize and then train
                the reservoir
            sync_steps (int): How many steps to use for synchronization before
                the prediction starts
            reg_param (float): weight for the Tikhonov-regularization term
            w_in_scale (float): maximum absolute value of the (random) w_in
                elements
            w_in_sparse (bool): If true, creates w_in such that one element in
                each row is non-zero (Lu,Hunt, Ott 2018)
            w_in_orderd (bool): If true and w_in_sparse is true, creates w_in
                such that elements are ordered by dimension and number of 
                elements for each dimension is equal (as far as possible)
            w_in_no_update (bool): If true and the input matrix W_in does
                already exist from a previous training run, W_in does not get
                updated, regardless of all other parameters concerning it.
            act_fct_flag (int_or_str): Specifies the activation function to be
                used during training (and prediction). Possible flags and their
                synonyms are:

                - "tanh_simple", "simple"
                - "tanh_bias"
                -
            bias_scale (float): Bias to be used in some activation functions
            mix_ratio (float): Ratio of normal tanh vs squared tanh activation
                functions if act_fct_flag "mixed" is chosen
            save_r (bool): If true, saves r(t) internally
            save_input (bool): If true, saves the input data internally
            w_out_fit_flag (str): Type of nonlinear transformation applied to
                the reservoir states r to be used during the fit (and future
                prediction)
            loc_nbhd (np.ndarray): The local neighborhood used for the
                generalized local states approach. For more information, please
                see the docs.
        """
        self._reg_param = reg_param
        self._loc_nbhd = loc_nbhd

        x_dim = x_train.shape[1]
        if self._w_in is not None and w_in_no_update:
            if not self._x_dim == x_dim:
                raise Exception(
                    f'the x_dim specified in create_input_matrix does not match the data x_dim: {self._x_dim} vs {x_dim}')
        else:
            self.create_input_matrix(x_dim, w_in_scale=w_in_scale, w_in_sparse=w_in_sparse, w_in_ordered=w_in_ordered)

        if self._act_fct_flag is None:
            self.set_activation_function(act_fct_flag=act_fct_flag,
                                         bias_scale=bias_scale)

        if sync_steps != 0:
            x_sync = x_train[:sync_steps]
            x_train = x_train[sync_steps:]
            self.synchronize(x_sync)
        else:
            x_sync = None

        if save_input:
            self._x_train_sync = x_sync
            self._x_train = x_train

        if save_r:
            self._r_train, self._r_train_gen = self._train_synced(x_train,
                                                                  w_out_fit_flag=w_out_fit_flag,
                                                                  save_x_train_pred=save_x_train_pred)
        else:
            self._train_synced(x_train, w_out_fit_flag=w_out_fit_flag, save_x_train_pred=save_x_train_pred)

    def run_loop(self, steps, save_r=False):
        """
        Run the autonomous reservoir dynamics in a loop
        """
        y_pred = np.zeros((steps, self._x_dim))

        if save_r:
            self._r_pred = np.zeros((steps, self._network.shape[0]))
            self._r_pred_gen = self._r_to_generalized_r(self._r_pred)

            for i in range(steps):
                y_pred[i, :] = self._predict_step()
                self._r_pred[i] = self._last_r
                self._r_pred_gen[i] = self._last_r_gen
        else:
            for i in range(steps):
                y_pred[i, :] = self._predict_step()
        return y_pred

    def predict(self, x_pred=None, sync_steps=0, pred_steps=None,
                save_r=False, save_input=False):
        """ Synchronize the reservoir, then predict the system evolution

        Changes self._last_r and self._last_r_gen to stay synchronized to the
        new system state

        Args:
            x_pred (np.ndarray): Input data used to synchronize the reservoir,
                and then used as comparison for the prediction by being returned
                as y_pred in the output
            sync_steps (int): How many steps to use for synchronization before
                the prediction starts
            pred_steps (int): How many steps to predict
            save_r (bool): If true, saves r(t) internally
            save_input (bool): If true, saves the input data internally

        Returns:
            tuple: 2-element tuple containing:

            - **y_pred** (*np.ndarray*): Predicted future states
            - **y_test** (*np.ndarray_or_None*): Data taken from the input to
              compare the prediction with. If the prediction were
              "perfect" y_pred and y_test would be equal. Be careful
              though, y_test might be shorter than y_pred, or even None,
              if pred_steps is not None

        """

        if pred_steps is None:
            pred_steps = x_pred.shape[0] - sync_steps - 1

        # if x_pred is None:
        #
        if len(x_pred) > sync_steps + pred_steps + 1:
            x_pred = x_pred[:sync_steps + pred_steps + 1]

        # Automatically generates a y_test to compare the prediction against, if
        # the input data is longer than the number of synchronization tests
        if sync_steps == 0:
            x_sync = None
            y_test = x_pred[1:]
        elif sync_steps <= x_pred.shape[0]:
            x_sync = x_pred[:sync_steps]
            y_test = x_pred[sync_steps + 1:]
        else:
            x_sync = x_pred[:-1]
            y_test = None

        if save_input:
            self._x_pred_sync = x_sync
            self._y_test = y_test

        if x_sync is not None:
            self.synchronize(x_sync)

        self.logger.debug('Start Prediction')

        self._y_pred = np.zeros((pred_steps, x_pred.shape[1]))

        self._y_pred[0] = self._predict_step(x_pred[sync_steps])

        if save_r:
            self._r_pred = np.zeros((pred_steps, self._network.shape[0]))
            self._r_pred_gen = self._r_to_generalized_r(self._r_pred)
            self._r_pred[0] = self._last_r
            self._r_pred_gen[0] = self._last_r_gen

            for t in range(pred_steps - 1):
                self._y_pred[t + 1] = self._predict_step(self._y_pred[t])

                self._r_pred[t + 1] = self._last_r
                self._r_pred_gen[t + 1] = self._last_r_gen

        else:
            for t in range(pred_steps - 1):
                self._y_pred[t + 1] = self._predict_step(self._y_pred[t])

        return self._y_pred, y_test

    def get_network(self, as_array=True):
        """ Returns the network used as reservoir
        Args:
            as_array (bool): if True return network as dense numpy array, else return as scipy.sparse matrix
        Returns:
            csr_matrix: network saved as scipy.sparse matrix
            or
            np.ndarray: network saved as np.ndarray matrix

        """
        if as_array:
            return self._network.toarray()
        else:
            return self._network

    def get_network_parameters(self):
        """ Returns all network parameters

        Returns:
            dict: Dictionary of network parameters
        """

        param_dict = {"n_dim": self._n_dim,
                      "n_rad": self._n_rad,
                      "n_avg_deg": self._n_avg_deg,
                      "n_edge_prob": self._n_edge_prob,
                      "n_type_flag": self._n_type_flag}

        return param_dict

    def summary(self):
        out_msg = ""
        layer_wall = "=====================\n"
        description_wall = "- - - - - - - - - - -\n"

        out_msg += "INPUT: \n"
        out_msg += f"x_dim: {self._x_dim}\n"

        out_msg += layer_wall
        out_msg += "INPUT-RESERVOIR COUPLING W_in: \n"
        try:
            out_msg += f"W_in: {self._w_in.shape}\n"
        except Exception as e:
            out_msg += f"W_in: {self._w_in}\n"
        out_msg += description_wall
        out_msg += f"w_in_scale: {self._w_in_scale}, w_in_sparse:" \
                   f" {self._w_in_sparse}, w_in_ordered: {self._w_in_ordered}\n"

        out_msg += layer_wall
        out_msg += "RESERVOIR: \n"
        out_msg += description_wall
        out_msg += "Network: \n"
        out_msg += f"n_dim: {self._n_dim}\n"
        out_msg += f"n_rad: {self._n_rad}, n_avg_deg: {self._n_avg_deg}, n_type_flag: {self._n_type_flag_str}, \n"
        out_msg += description_wall
        out_msg += "Dynamics: \n"
        out_msg += f"act_fct_flag: {self._act_fct_flag_str}, " \
                   f"bias_scale: {self._bias_scale}, leak_factor: {self._leak_fct}\n"
        # out_msg += "Update equation: r(i+1) = leak_fct * r(i) + (1-leak_fct) * act_fct(W_in * x(i) + W * r(i))\n"
        out_msg += layer_wall
        out_msg += "NON-LINEAR TRANSFORM OF RESERVOIR STATE: \n"
        out_msg += f"w_out_fit_flag: {self._w_out_fit_flag_str}\n"
        out_msg += layer_wall
        out_msg += "RESERVOIR-OUTPUT COUPLING W_out: \n"
        try:
            out_msg += f"W_out: {self._w_out.shape}\n"
        except Exception as e:
            out_msg += f"W_out: {self._w_out}\n"

        print(out_msg)

    def get_w_in(self, ):
        """ Returns the input matrix w_in

        Returns:
            np.ndarray: w_in

        """

        return self._w_in

    def get_w_out(self, ):
        """ Returns the output matrix w_out

        Returns:
            np.ndarray: w_out

        """

        return self._w_out

    # def get_w_in_parameters(self, ):
    #     raise Exception("Not yet implemented")
    #
    # def get_activation_function(self, ):
    #     raise Exception("Not yet implemented")
    #
    # def get_training(self, ):
    #     raise Exception("Not yet implemented")
    #
    # def get_prediction(self, ):
    #     raise Exception("Not yet implemented")

    def get_instance_version(self):
        """ Returns the rescomp package version used to create the class instance

        Returns:
            str: Rescomp package version

        """
        return self._rescomp_version

    def to_pickle(self, path, compression="infer",
                  protocol=pickle.HIGHEST_PROTOCOL):
        """ Pickle (serialize) object to file.

        Disables logging as logging handlers can not be pickled.
        Uses pandas functions internally.

        Args:
            path (str): File path where the pickled object will be stored.
            compression ({'infer', 'gzip', 'bz2', 'zip', 'xz', None}):
                A string representing the compression to use in the output file.
                By default, infers from the file extension in specified path.
            protocol (int): Int which indicates which protocol should be used by
                the pickler. The possible values are 0, 1, 2, 3, 4. A negative
                value for the protocol parameter is equivalent to setting its
                value to HIGHEST_PROTOCOL.

        """

        self.logger.debug("Save to file %s, turn off internal logger to do so" % path)

        self.set_console_logger("off")
        self.set_file_logger("off", None)

        pandas.io.pickle.to_pickle(self, path, compression=compression,
                                   protocol=protocol)


class ESNWrapper(ESN):
    """ Convenience Wrapper for the ESN class

    For all normal usage of the rescomp package, this is the class to use.

    """

    def __init__(self):
        super().__init__()
        self.logger.debug("Create ESNWrapper instance")

    def train_and_predict(self, x_data, train_sync_steps, train_steps,
                          pred_sync_steps=0, pred_steps=None, disc_steps=0, **kwargs):
        """ Train, then predict the evolution directly following the train data

        Args:
            x_data (np.ndarray): Data used for synchronization, training and
                prediction (start and comparison)
            train_sync_steps (int): Steps to synchronize the reservoir with
                before the 'real' training begins
            train_steps (int): Steps to use for training and fitting w_in
            pred_sync_steps (int): steps to sync the reservoir with before
                prediction
            pred_steps (int): How many steps to predict the evolution for
            **kwargs: further arguments passed to :func:`~esn.ESN.train` and
                :func:`~esn.ESN.predict`

        Returns:
            tuple: 2-element tuple containing:

            - **y_pred** (*np.ndarray*): Predicted future states
            - **y_test** (*np.ndarray_or_None*): Data taken from the input to
              compare the prediction with. If the prediction were
              "perfect" y_pred and y_test would be equal. Be careful
              though, y_test might be shorter than y_pred, or even None,
              if pred_steps is not None

        """
        x_train, x_pred = utilities.train_and_predict_input_setup(
            x_data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_sync_steps=pred_sync_steps,
            pred_steps=pred_steps)

        train_kwargs = utilities._remove_invalid_args(self.train, kwargs)
        predict_kwargs = utilities._remove_invalid_args(self.predict, kwargs)

        self.train(x_train, train_sync_steps, **train_kwargs)

        y_pred, y_test = self.predict(x_pred, sync_steps=pred_sync_steps,
                                      pred_steps=pred_steps, **predict_kwargs)

        return y_pred, y_test

    def create_architecture(self, n_dim, x_dim, n_rad=0.1, n_avg_deg=6.0, w_in_scale=1.0,
                            n_type_flag="erdos_renyi", network_creation_attempts=10,
                            w_in_sparse=True, w_in_ordered=False, w_out_fit_flag="simple",
                            seed=42, activation_function="tanh_simple", leak_fct=0.0, res_state=None,
                            w_out=None, custom_act_fct=None):
        if seed is not None:
            np.random.seed(seed)
        self.create_network(n_dim=n_dim, n_rad=n_rad, n_avg_deg=n_avg_deg,
                            n_type_flag=n_type_flag, network_creation_attempts=network_creation_attempts)
        self.create_input_matrix(x_dim, w_in_scale=w_in_scale, w_in_sparse=w_in_sparse, w_in_ordered=w_in_ordered)
        self.set_activation_function(activation_function, leak_fct=leak_fct, custom_act_fct=custom_act_fct)
        self.reset_res_state(res_state=res_state)
        self.set_w_out_fit_flag(w_out_fit_flag)
        self.set_w_out(w_out)

        # run test loop:
        try:
            self._predict_step()
            self.reset_res_state(res_state=res_state)
            self.logger.debug("ESN can run in loop")
        except Exception as e:
            print(e)
            self.logger.debug("Some ESN parameters missing, can not run loop")


class ESNGenLoc(utilities._ESNLogging):
    """ Generalized Local State Implementation of RC

    For details on the how and why behind this implementation, please see the
    general_local_states_example in bin and/or the package documentation.

    Idea behind the Local Neighborhoods matrix implementation:
    Human readable and plottable locality neighborhoods where each entry
    specifies the nature of a corresponding dimension.
    A simple example might look like this:
        [1, 2, 1, 0] \n
        [0, 1, 2, 1] \n
        [2, 1, 1, 2]
    where 0 means "not part of the neighborhood", 1 means "part of the
    neighborhood, but not a core" and 2 means "core dimension"
    Each row is a "neighborhood" specifying which dimensions are
    grouped together as input for each ESN instance.
    As such there can only be one 2 per input dimension (column) as
    otherwise combining the different neighborhood's predictions after
    each prediction step is not possible.
    In theory there can be no 2 in a dimension/column though. In that case
    the corresponding dimension just doesn't appear in the prediction output.

    The code is written under the assumption that all internal ESNs have
    the same hyperparameters. This does not necessarily need to be the case
    though, and a more generalized implementation would allow it.
    """

    def __init__(self):
        """ Same idea as for the ESN Wrapper Class """
        super().__init__()

        self._esn_instances = []

        self._nr_nbhds = None
        self._loc_nbhds = None
        self._train_core_only = None

    def create_network(self, **kwargs):
        """ Same idea as for the ESN Wrapper Class """
        esn = ESNWrapper()
        esn.create_network(**kwargs)
        self._esn_instances.append(esn)

    def train(self, x_train, sync_steps, loc_nbhds=None, train_core_only=True,
              ESN_train_kwargs=None):
        """ Same idea as for the ESN Wrapper Class """

        self._loc_nbhds = loc_nbhds

        self._nr_nbhds = loc_nbhds.shape[0]
        self._train_core_only = train_core_only

        if ESN_train_kwargs is None: ESN_train_kwargs = {}

        self.logger.debug("Start locality training with %d neighborhoods" %
                          self._nr_nbhds)

        self.logger.debug("Deepcopy initial ESN instance for each "
                          "Neighborhood. Reservoir network matrix are shallow "
                          "copies though.")

        self._esn_instances[0].set_console_logger("off")
        self._esn_instances[0].set_file_logger("off", None)
        self._esn_instances = [deepcopy(self._esn_instances[0]) for i in range(self._nr_nbhds)]

        self.logger.debug("Start Training of Neighborhood 1/%d" % self._nr_nbhds)
        loc_x_train = self._get_local_x(x_train, 0)
        esn = self._esn_instances[0]
        if train_core_only:
            loc_nbhd = self._get_nbhd_cut_off_zeros(0)
        else:
            loc_nbhd = None
        esn.train(loc_x_train, sync_steps, loc_nbhd=loc_nbhd, **ESN_train_kwargs)

        # This sets all the reservoirs to be the same object in memory. This is
        # more efficient, but less general and would need to be reworked if one
        # wanted to support different reservoir configurations.
        # The same is true for the input matrices.
        for nbhd_index in range(1, self._nr_nbhds):
            self._esn_instances[nbhd_index]._network = self._esn_instances[0]._network
            # NOTE This next line throws an error if there are neighborhoods with
            #  more or less elements than the 0th neighborhood. I keep it here
            #  for now to reproduce old results and for a slight bit more memory
            #  efficiency.
            #  Also note the necessary w_in_no_update=True in the loop below.
            self._esn_instances[nbhd_index]._w_in = self._esn_instances[0]._w_in

        for nbhd_index in range(1, self._nr_nbhds):
            self.logger.debug("Start Training of Neighborhood %d/%d" %
                              (nbhd_index + 1, self._nr_nbhds))
            loc_x_train = self._get_local_x(x_train, nbhd_index)
            esn = self._esn_instances[nbhd_index]
            if train_core_only:
                loc_nbhd = self._get_nbhd_cut_off_zeros(nbhd_index)
            else:
                loc_nbhd = None
            esn.train(loc_x_train, sync_steps, loc_nbhd=loc_nbhd,
                      w_in_no_update=True, **ESN_train_kwargs)
            gc.collect()

    def _get_local_x(self, x, nbhd_index):
        nbhd = self._get_nbhd(nbhd_index)
        x_ndim = x.ndim
        if x_ndim == 2:
            loc_x = np.squeeze(x[:, np.nonzero(nbhd)], 1)
        elif x_ndim == 1:
            loc_x = x[np.nonzero(nbhd)]
        else:
            raise Exception("x.ndim must be either 1 or 2")

        return loc_x

    def _get_nbhd_cut_off_zeros(self, nbhd_index):
        nbhd = self._get_nbhd(nbhd_index)
        nbhd_cut_of_zeros = nbhd[np.nonzero(nbhd)]
        return nbhd_cut_of_zeros

    def _get_nbhd(self, nbhd_index):
        # Each row of self._loc_nbhds is a neighborhood
        nbhd = self._loc_nbhds[nbhd_index]
        return nbhd

    def predict(self, x_pred, sync_steps, pred_steps=None, ESN_sync_kwargs=None,
                **kwargs):
        """ Same idea as for the ESN Wrapper Class """
        # NOTE This function is very inefficiently written, but it doesn't
        #  really matter, as the prediction is very much not the bottleneck
        #  anyway. Still worth improving though.

        if ESN_sync_kwargs is None: ESN_sync_kwargs = {}

        self.logger.debug('Start local prediction')

        # NOTE Much copied from ESN class -> Better way than code duplication?
        if pred_steps is None:
            pred_steps = x_pred.shape[0] - sync_steps - 1

        if len(x_pred) > sync_steps + pred_steps + 1:
            x_pred = x_pred[:sync_steps + pred_steps + 1]

        if sync_steps == 0:
            x_sync = None
            y_test = x_pred[1:]
        elif sync_steps <= x_pred.shape[0]:
            x_sync = x_pred[:sync_steps]
            y_test = x_pred[sync_steps + 1:]
        else:
            x_sync = x_pred[:-1]
            y_test = None

        # Synchronize
        if x_sync is not None:
            self.synchronize(x_sync)

        # If pred_steps == 0, this function only syncs, and doesn't actually
        # predict anything
        if pred_steps != 0:
            # Predict Step by step
            # NOTE The 2nd line, as well as the range for t is a hack to avoid code
            #  duplication in the prediction loop. It doesn't cause problems as
            #  _y_pred[-1] will be overwritten in the last prediction step anyway.
            #  Still probably error prone though
            y_pred = np.zeros((pred_steps, x_pred.shape[1]))
            y_pred[-1] = deepcopy(x_pred[sync_steps])
            for t in range(-1, pred_steps - 1):

                if t % 1000 == 0:
                    self.logger.debug('Prediction for %d/%d steps done' % (t, pred_steps))

                y_pred[t + 1] = self._predict_step(y_pred[t])

            self.logger.debug('Prediction for %d/%d steps done' %
                              (pred_steps, pred_steps))

            # sync in parallel, same as for the train from above
            # pred every step at a time
            return y_pred, y_test
        else:
            return None, None

    def synchronize(self, x_sync, **kwargs):
        """ Same idea as for the ESN Wrapper Class """
        # NOTE This function is very inefficiently written, but it doesn't
        #  really matter, as the synchronization is very much not the bottleneck
        #  anyway. Still improvement worthy though.

        self.logger.debug('Start syncing')

        for nbhd_index in range(self._nr_nbhds):
            loc_x_sync = self._get_local_x(x_sync, nbhd_index)

            esn = self._esn_instances[nbhd_index]
            esn.synchronize(loc_x_sync, **kwargs)

    def _predict_step(self, x=None):
        # NOTE I think this for loop can, i.g. not be translated to completely
        #  vectorized numpy code as i.g. loc_y_pred and loc_nbhd_cut_of_zeros
        #  can have different sizes for different neighborhood numbers

        # If x isn't given, calculate it from the last r states as the x we are
        # looking for is just the y from the last step
        if x is None:
            x = np.zeros(shape=self._loc_nbhds.shape[1])
            for nbhd_index in range(self._nr_nbhds):
                esn = self._esn_instances[nbhd_index]

                # NOTE Predicting a step advances the reservoir state by a step,
                #  which we don't want, hence if we call esn._predict step we'd
                #  need to reset it afterwards, like this:
                #  last_r = esn._last_r
                #  loc_x = esn._predict_step()
                #  esn._last_r = last_r
                #  This is a very hacky and bad way to do this, which is why
                #  we copy the relevant line from the prediction code here
                #  instead. This is, of course, also bad, but not
                #  doable otherwise without a rewrite of _ESNCore
                loc_x = esn._w_out @ esn._r_to_generalized_r(esn._last_r)

                # NOTE: Copied this from the esn._predict_step() function
                #  because the padding is needed below. This is bad bad code
                temp = np.empty(esn._loc_nbhd.shape)
                temp[:] = np.nan
                temp[esn._loc_nbhd == 2] = loc_x
                loc_x = temp

                nbhd = self._get_nbhd(nbhd_index)
                nbhd_cut_of_zeros = self._get_nbhd_cut_off_zeros(nbhd_index)

                x[nbhd == 2] = loc_x[nbhd_cut_of_zeros == 2]

        y = np.zeros(shape=self._loc_nbhds.shape[1])
        for nbhd_index in range(self._nr_nbhds):
            esn = self._esn_instances[nbhd_index]

            loc_x_pred_step = self._get_local_x(x, nbhd_index)
            loc_y_pred_step = esn._predict_step(loc_x_pred_step)

            nbhd = self._get_nbhd(nbhd_index)
            nbhd_cut_of_zeros = self._get_nbhd_cut_off_zeros(nbhd_index)

            y[nbhd == 2] = loc_y_pred_step[nbhd_cut_of_zeros == 2]

        return y

    def train_and_predict(self, x_data, train_sync_steps, train_steps,
                          loc_nbhds=None, pred_sync_steps=0, pred_steps=None,
                          disc_steps=0, **kwargs):
        """ Same idea as for the ESN Wrapper Class """
        # NOTE: Code duplication -> Bad -> Template functions or smth?

        x_train, x_pred = utilities.train_and_predict_input_setup(
            x_data, disc_steps=disc_steps, train_sync_steps=train_sync_steps,
            train_steps=train_steps, pred_sync_steps=pred_sync_steps,
            pred_steps=pred_steps)

        self_train_kwargs = utilities._remove_invalid_args(
            self.train, kwargs)
        ESN_train_kwargs = utilities._remove_invalid_args(
            ESNWrapper.train, kwargs)
        self_predict_kwargs = utilities._remove_invalid_args(
            self.predict, kwargs)
        ESN_sync_kwargs = utilities._remove_invalid_args(
            ESNWrapper.predict, kwargs)

        # make sure that all kwargs are actually passed to a function somewhere
        all_passed_kwargs = deepcopy(self_train_kwargs)
        all_passed_kwargs.update(ESN_train_kwargs)
        all_passed_kwargs.update(self_predict_kwargs)
        all_passed_kwargs.update(ESN_sync_kwargs)
        if not all(item in all_passed_kwargs.items() for item in kwargs.items()):
            raise Exception("Not all arguments passed to functions during "
                            "train_and_predict()!")

        self.train(x_train, train_sync_steps, loc_nbhds,
                   ESN_train_kwargs=ESN_train_kwargs, **self_train_kwargs)
        y_pred, y_test = self.predict(x_pred, sync_steps=pred_sync_steps,
                                      pred_steps=pred_steps, ESN_sync_kwargs=ESN_sync_kwargs,
                                      **self_predict_kwargs)

        return y_pred, y_test

    def to_pickle(self, path, compression="infer",
                  protocol=pickle.HIGHEST_PROTOCOL):
        """ Pickle (serialize) object to file.

        Disables logging as logging handlers can not be pickled.
        Uses pandas functions internally.

        Args:
            path (str): File path where the pickled object will be stored.
            compression ({'infer', 'gzip', 'bz2', 'zip', 'xz', None}):
                A string representing the compression to use in the output file.
                By default, infers from the file extension in specified path.
            protocol (int): Int which indicates which protocol should be used by
                the pickler. The possible values are 0, 1, 2, 3, 4. A negative
                value for the protocol parameter is equivalent to setting its
                value to HIGHEST_PROTOCOL.

        """

        log_file_path = self._log_file_path
        console_log_level = self._console_log_level
        file_log_level = self._file_log_level

        self.logger.debug("Save LocalESN to file %s" % path)

        self.set_console_logger("off")
        self.set_file_logger("off", None)

        pandas.io.pickle.to_pickle(self, path, compression=compression,
                                   protocol=protocol)

        self.set_console_logger(console_log_level)
        self.set_file_logger(file_log_level, log_file_path)

    def get_last_r_states(self):
        """
        Returns:

        """
        last_r_states = []
        for esn in self._esn_instances:
            last_r_states.append(esn._last_r)
        last_r_states = np.array(last_r_states)
        return last_r_states

    def set_last_r_states(self, r_states):
        for i in range(r_states.shape[0]):
            last_r = r_states[0]
            esn = self._esn_instances[0]
            esn._last_r = last_r


class ESNHybrid(ESNWrapper):
    """
    Added by Dennis Duncan: Knowledge-based Hybrid Reservoir Computing:
    Combining Knowledge based models with purely data-driven reservoir computing

    Following: (1) ArXiv ID: 1803.04779
    """

    def __init__(self):
        super().__init__()
        self.logger.debug("Create ESNWrapper instance")
        self.model = None

        self.add_model_to_output = None #If True, the model-prediction is included into the output layer. See (1)
        self.add_model_to_input = None #If True the input to the reservoir is both the model(x) and x. See (1)
        self.gamma = None # Fraction of the reservoir nodes connected exclusivly to the raw input, Only has influence if add_model_to_input = True

    def set_model(self, model, add_model_to_output = False,  add_model_to_input = False, gamma = 0.5):
        '''
        :param model:
        :return:
        '''
        self.model = model
        self.add_model_to_output = add_model_to_output
        self.add_model_to_input = add_model_to_input
        self.gamma = gamma

    def model_array(self, x):
        u_train = np.zeros(x.shape)
        for i, x_i in enumerate(x):
            u_train[i] = self.model(x_i)
        return u_train

    def _fit_w_out(self, x_train, r):
        """
        --DD: Altered to work for hybrid reservoir computing--

        Fit the output matrix self._w_out after training

        Uses linear regression and Tikhonov regularization.

        Args:
            y_train (np.ndarray): Desired prediction from the reservoir states
            r (np.ndarray): reservoir states
        Returns:
            np.ndarray: r_gen, generalized nonlinear transformed r

        """

        # DD: In the Hybrid RC, the whole x_train has to be used for the training,
        # thus _train_synced and _fit_w_out have to be overwritten
        y_train = x_train[1:]

        self.logger.debug('Fit _w_out according to method %s' %
                          str(self._w_out_fit_flag))

        r_gen = self._r_to_generalized_r(r)

        # --DD: Stack the model-based prediction on top of r_gen--
        if self.add_model_to_output:
            u_train = self.model_array(x_train[:-1])
            r_gen = np.concatenate((r_gen, u_train), axis = 1)

        # If we are using local states we only want to use the core dimensions
        # for the fit of W_out, i.e. the dimensions where the corresponding
        # locality matrix is 2
        # TODO: localstates and hybrid not combined yet
        if self._loc_nbhd is None:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train).T
        else:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train[:, self._loc_nbhd == 2]).T

        return r_gen

    def _predict_step(self, x=None):
        """ Predict a single time step

        Assumes a synchronized reservoir.
        Changes self._last_r and self._last_r_gen to stay synchronized to the
        new system state y

        Args:
            x (np.ndarray): input for the d-dim. system, shape (d,). If x is
            None the internal reservoir states will be used to generate x
            using w_out

        Returns:
            np.ndarray: y, the next time step as predicted from last_x, _w_out
            and _last_r, shape (d,)

        """

        if x is None:
            x = self._w_out @ self._r_to_generalized_r(self._last_r)

        if self.add_model_to_input:
            model_x = self.model(x)
            x_in = np.concatenate((x, model_x), axis = 0)
        else:
            x_in = x

        self._last_r = self._act_fct(x_in, self._last_r)
        self._last_r_gen = self._r_to_generalized_r(self._last_r)

        last_r_gen = self._last_r_gen

        if self.add_model_to_output:
            if not self.add_model_to_input: # dont calculate it twice to save time
                model_x = self.model(x)
            last_r_gen = np.concatenate((last_r_gen, model_x))

        y = self._w_out @ last_r_gen

        if self._loc_nbhd is not None:
            temp = np.empty(self._loc_nbhd.shape)
            temp[:] = np.nan
            temp[self._loc_nbhd == 2] = y
            y = temp

        return y

    def _create_w_in(self):
        """  Create the input matrix w_in

        Specification done via protected members
        TODO: All behaviours should be tested more thoroughly
        """
        self.logger.debug("Create w_in")

        if self.add_model_to_input:
            x_dim = 2 * self._x_dim
            nr_res_nodes_connected_to_raw_in = int(self.gamma * self._n_dim)
            nr_res_nodes_connected_to_model_in = self._n_dim - nr_res_nodes_connected_to_raw_in
            # print("Actual percentage of Reservoir nodes connected to raw input: ", np.round(nr_res_nodes_connected_to_raw_in*100/self._n_dim, 1))
        else:
            x_dim = self._x_dim
            nr_res_nodes_connected_to_raw_in = self._n_dim

        if self._w_in_sparse and not self._w_in_ordered:
            self._w_in = np.zeros((self._n_dim, x_dim))

            nodes_connected_to_raw = np.random.choice(np.arange(self._n_dim), size=nr_res_nodes_connected_to_raw_in, replace=False)
            nodes_connected_to_raw = np.sort(nodes_connected_to_raw)

            for index in nodes_connected_to_raw:
                random_x_coord = np.random.choice(np.arange(self._x_dim))
                self._w_in[index, random_x_coord] = np.random.uniform(
                    low=-self._w_in_scale,
                    high=self._w_in_scale)

            if self.add_model_to_input:
                nodes_connected_to_model = np.delete(np.arange(self._n_dim), nodes_connected_to_raw)

                for index in nodes_connected_to_model:
                    random_x_coord = np.random.choice(np.arange(self._x_dim))
                    self._w_in[index, random_x_coord + self._x_dim] = np.random.uniform(
                        low=-self._w_in_scale,
                        high=self._w_in_scale)

        elif self._w_in_sparse and self._w_in_ordered:
            # DD: hybrid model missing here sofar
            # new:
            raw_input_node_indices = np.arange(0, self._x_dim)
            res_node_ind_connected_to_raw = np.arange(0, nr_res_nodes_connected_to_raw_in)

            self._w_in = np.zeros((self._n_dim, x_dim))

            dim_wise = np.array([int(nr_res_nodes_connected_to_raw_in / self._x_dim)] * self._x_dim)
            dim_wise[:nr_res_nodes_connected_to_raw_in % self._x_dim] += 1

            c = 0
            for d in raw_input_node_indices:
                nr_of_connections_to_dim = dim_wise[d]
                for con in range(nr_of_connections_to_dim):
                    reservoir_node = res_node_ind_connected_to_raw[c]
                    c+=1
                    self._w_in[reservoir_node, d] = np.random.uniform(
                        low=-self._w_in_scale,
                        high=self._w_in_scale)  # maps input values to reservoir

            if self.add_model_to_input:
                model_input_node_indices = np.arange(self._x_dim, 2*self._x_dim)
                res_node_ind_connected_to_model = np.arange(nr_res_nodes_connected_to_raw_in, nr_res_nodes_connected_to_model_in + nr_res_nodes_connected_to_raw_in)

                dim_wise = np.array([int(nr_res_nodes_connected_to_model_in / self._x_dim)] * self._x_dim)
                dim_wise[:nr_res_nodes_connected_to_model_in % self._x_dim] += 1
                c = 0
                for i, d in enumerate(model_input_node_indices):
                    nr_of_connections_to_dim = dim_wise[i]
                    for con in range(nr_of_connections_to_dim):
                        reservoir_node = res_node_ind_connected_to_model[c]
                        c += 1
                        self._w_in[reservoir_node, d] = np.random.uniform(
                            low=-self._w_in_scale,
                            high=self._w_in_scale)  # maps input values to reservoir
        else:
            self._w_in = np.random.uniform(low=-self._w_in_scale,
                                           high=self._w_in_scale,
                                           size=(self._n_dim, x_dim))

    def synchronize(self, x, save_r=False):
        """ Synchronize the reservoir state with the input time series

        This is usually done automatically in the training and prediction
        functions.
        DD: Add model input

        Args:
            x (np.ndarray): Input data to be used for the synchronization,
                shape (T, d)
            save_r (bool): If true, saves and returns r

        Returns:
            np.ndarray_or_None: All r states if save_r is True, None if False

        """
        self.logger.debug('Start syncing the reservoir state')

        if self.add_model_to_input:
            x = np.concatenate((x, self.model_array(x)), axis = 1)
        if self._last_r is None:
            self._last_r = np.zeros(self._network.shape[0])

        if save_r:
            r = np.zeros((x.shape[0], self._network.shape[0]))
            r[0] = self._act_fct(x[0], self._last_r)
            for t in np.arange(x.shape[0] - 1):
                r[t+1] = self._act_fct(x[t + 1], r[t])
            self._last_r = deepcopy(r[-1])
            return r
        else:
            for t in np.arange(x.shape[0]):
                self._last_r = self._act_fct(x[t], self._last_r)
            return None

    def set_w_out_to_only_model(self):
        '''
        Just a experimental function that wires all connection in w_out to only read
        the model based prediction (if output_hybrid)
        :return:
        '''
        if self.add_model_to_output:
            # modify _w_out:
            self.logger.debug("Rewire all connections to model: Change W_out")
            n_dim_mod = self._n_dim + self._x_dim
            x_dim = self._x_dim
            matrix = np.zeros((x_dim, n_dim_mod))
            for i in range(0, x_dim):
                to_add = np.zeros(x_dim)
                to_add[i] = 1
                matrix[:, n_dim_mod - x_dim + i] = to_add
            self._w_out = matrix
        else:
            self.logger.debug("Error: This function only has an effect if add_model_to_output is True")


class ESNDifference(ESNWrapper):
    """
    Added by Dennis Duncan: A RC model that trains on the difference:  [u_(i+1)-u_(i)]/dt

    Notes:
        - Either change _fit_w_out or change code where _fit_w_out is called
        - same for _train_synced
        - change _predict_step
        - _train_synced is called in Train
    """

    def __init__(self, sampling_time=1):
        super().__init__()
        self.logger.debug("Create ESNWrapper instance")

        self.sampling_time = sampling_time  # the time between two points of the input timeseries

        self._last_input = None

    def _fit_w_out(self, x_train, r, save_x_train_pred=False):
        """ Fit the output matrix self._w_out after training
        Trains on the difference
        Uses linear regression and Tikhonov regularization.

        Args:
            x_train (np.ndarray): get y_train via x_train[1:], y_train = Desired prediction from the reservoir states
            r (np.ndarray): reservoir states
        Returns:
            np.ndarray: r_gen, generalized nonlinear transformed r

        """

        y_train = (x_train[1:] - x_train[:-1])/self.sampling_time
        # y_train = x_train[1:]

        self.logger.debug('Fit _w_out according to method %s' %
                          str(self._w_out_fit_flag))

        r_gen = self._r_to_generalized_r(r)

        # If we are using local states we only want to use the core dimensions
        # for the fit of W_out, i.e. the dimensions where the corresponding
        # locality matrix == 2
        if self._loc_nbhd is None:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train).T
        else:
            self._w_out = np.linalg.solve(
                r_gen.T @ r_gen + self._reg_param * np.eye(r_gen.shape[1]),
                r_gen.T @ y_train[:, self._loc_nbhd == 2]).T

        if save_x_train_pred:
            print(r_gen.shape)
            print(self._w_out.shape)
            self.x_train_pred = (self._w_out @ r_gen.T).T

        return r_gen

    def _predict_step(self, x=None):
        """ Predict a single time step

        Assumes a synchronized reservoir.
        Changes self._last_r and self._last_r_gen to stay synchronized to the
        new system state y

        Args:
            x (np.ndarray): input for the d-dim. system, shape (d,). If x is
            None the internal reservoir states will be used to generate x
            using w_out

        Returns:
            np.ndarray: y, the next time step as predicted from last_x, _w_out
            and _last_r, shape (d,)
        """

        if x is None:
            x = self._w_out @ self._r_to_generalized_r(self._last_r) * self.sampling_time + self._last_input  # add smth

        self._last_input = x
        self._last_r = self._act_fct(x, self._last_r)
        self._last_r_gen = self._r_to_generalized_r(self._last_r)

        y = self._w_out @ self._last_r_gen * self.sampling_time + self._last_input

        self._last_input = y

        # ignore this for this class:
        if self._loc_nbhd is not None:
            temp = np.empty(self._loc_nbhd.shape)
            temp[:] = np.nan
            temp[self._loc_nbhd == 2] = y
            y = temp
        return y

    # def _act_fct(self, x, r):
    #     self._last_input = x
    #     super()._act_fct(x, r)

    def synchronize(self, *args, **kwargs):
        self._last_input = args[0][-1]
        return super().synchronize(*args, **kwargs)



    def create_architecture(self, *args, **kwargs):
        x_dim = args[1]
        self._last_input = np.zeros(x_dim)
        super().create_architecture(*args, **kwargs)
