# -*- coding: utf-8 -*-
""" Measures and other analysis functions useful for RC """

import numpy as np
import scipy
import scipy.sparse
from scipy.signal import argrelextrema

# import matplotlib.pyplot as plt
import warnings
from . import utilities
from . import simulations


# TODO: there should be a utilities._SynonymDict() here
def rmse_over_time(pred_time_series, meas_time_series, normalization=None):
    """ Calculates the NRMSE over time,

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
        normalization (str_or_None_or_float): The normalization method to use.
            Possible are:

            - None: Calculates the pure, standard RMSE
            - "mean": Calulates RMSE divided by the entire, flattened
              meas_time_series mean
            - "std_over_time": Calulates RMSE divided by the entire
              meas_time_series' standard deviation in time of dimension.
              See Vlachas, Pathak et al. (2019) for details
            - "2norm": Uses the vector  2-norm of the meas_time_series averaged
              over time normalize the RMSE for each time step
            - "maxmin": Divides the RMSE by (max(meas) - min(meas))
            - float: Calulates the RMSE, then divides it by the given float

    Returns:
        np.ndarray: RMSE for each time step, shape (T,)

    """
    pred = pred_time_series
    meas = meas_time_series

    if normalization == "mean":
        normalization = np.mean(meas)
    if normalization == "std_over_time":
        mean_std_over_time = np.mean(np.std(meas, axis=0))
        normalization = mean_std_over_time
    if normalization == "2norm":
        # euclid_norms = np.linalg.norm(meas, axis=0)
        # normalization = np.mean(euclid_norms)
        pass
    if normalization == "maxmin":
        maxmin = np.max(meas) - np.min(meas)
        normalization = maxmin
        pass
    nrmse_list = []

    for i in range(0, meas.shape[0]):
        local_nrmse = rmse(pred[i: i + 1], meas[i: i + 1], normalization)
        nrmse_list.append(local_nrmse)

    return np.array(nrmse_list)


# NOTE: Removed due to ambiguity of normalization type
# def nrmse(pred_time_series, meas_time_series):
#     """ Calculates the NRME between two time series
#
#     Internally just calls rmse with normalized=True
#
#     Args:
#         pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
#         meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
#
#     Returns:
#         float: NRMSE
#     """
#     return rmse(pred_time_series, meas_time_series, normalized=True)

# TODO: there should be a utilities._SynonymDict() here
def rmse(pred_time_series, meas_time_series, normalization=None):
    """ Calculates the root mean squared error between two time series

    The time series must be of equal length and dimension

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
        normalization (str_or_None_or_float): The normalization method to use. Possible are:

            - None: Calculates the pure, standard RMSE
            - "mean": Calulates RMSE divided by the measured time series mean
            - "std_over_time": Calulates RMSE divided by the measured time
              series' standard deviation in time of dimension. See the NRSME
              definition of Vlachas, Pathak et al. (2019) for details
            - float: Calulates the RMSE, then divides it by the given float
            - "2norm": Uses the vector 2-norm of the meas_time_series to
              normalize the RMSE for each time step
            - "maxmin": Divides the RMSE by (max(meas) - min(meas))
            - "historic": Old, weird way to normalize the NRMSE, kept here
              purely for backwards compatibility. Don't use if you are not 100%
              sure that's what you want.
    Returns:
        float: RMSE or NRMSE

    """
    pred = pred_time_series
    meas = meas_time_series

    error = np.linalg.norm(pred - meas) / np.sqrt(meas.shape[0])

    if normalization is None:
        error = error
    elif normalization == "mean":
        error = error / np.mean(meas)
    elif normalization == "std_over_time":
        error = error / np.mean(np.std(meas, axis=0))
    elif normalization == "2norm":
        error = error / np.linalg.norm(meas)
    elif normalization == "maxmin":
        error = error / (np.max(meas) - np.min(meas))
    elif normalization == "historic":
        error = error / np.linalg.norm(meas) * np.sqrt(meas.shape[0])
    elif utilities._is_number(normalization):
        error = error / normalization
    else:
        raise Exception("Type of normalization not implemented")

    # if normalized:
    #     error = np.linalg.norm(pred - meas) \
    #             / np.linalg.norm(meas)
    # else:
    #     error = np.linalg.norm(pred - meas) \
    #             / np.sqrt(meas.shape[0])

    return error


def error_over_time(pred_time_series, meas_time_series, distance_measure="L2", normalization=None, remove_nans=True):
    """ Calculates a general error between two time series

    The time series must be of equal length and dimension

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
        distance_measure (str or function): The distance measure over space dimensions d used between pred and meas.
            Possible are:
            - If str:
                - "L2": L2 (euclidian) norm
                - "RMSE": RMSE norm. If selected, this function behaves the same as the "rmse_over_time" function
            - If function:
                - custom function that works like: distance = function(delta), the function has to operate on the
                    second dimension

        normalization (str_or_None_or_float): The normalization method to use. Possible are:

            - None: Calculates the pure, standard RMSE
            - "mean": Calulates RMSE divided by the measured time series mean
            - "std_over_time": Calulates RMSE divided by the measured time
              series' standard deviation in time of dimension. See the NRSME
              definition of Vlachas, Pathak et al. (2019) for details
            - float: Calulates the RMSE, then divides it by the given float
            - "2norm": Uses the vector 2-norm of the meas_time_series to
              normalize the RMSE for each time step
            - "maxmin": Divides the RMSE by (max(meas) - min(meas))
            - "historic": Old, weird way to normalize the NRMSE, kept here
              purely for backwards compatibility. Don't use if you are not 100%
              sure that's what you want.
        remove_nans: if True, turn all nans into np.inf. This is useful when comparisons
                    of error and number has to made, e.g. in valid_times_index
    Returns:
        (np.ndarray): error array, shape (T,):
    """

    pred = pred_time_series
    meas = meas_time_series

    if remove_nans:
        is_nan = np.isnan(pred)
        if is_nan.any():
            pred[is_nan] = np.inf
        is_nan = np.isnan(meas)
        if is_nan.any():
            meas[is_nan] = np.inf

    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=1)
    if len(meas.shape) == 1:
        meas = np.expand_dims(meas, axis=1)

    delta = pred-meas

    if type(distance_measure) == str:
        if distance_measure == "L2":
            distance = np.linalg.norm(delta, axis=1)
        elif distance_measure == "rmse":
            distance = np.linalg.norm(delta, axis=1)  # / np.sqrt(meas.shape[1])
        else:
            raise Exception("Type of distance_measure not implemented")
        # more norms can be added here
    else:
        distance = distance_measure(delta)

    if normalization is None:
        norm = 1
    elif normalization == "mean":
        norm = np.mean(meas)
    elif normalization == "std_over_time":
        norm = np.mean(np.std(meas, axis=0))
    elif normalization == "2norm":
        norm = np.linalg.norm(meas)
    elif normalization == "maxmin":
        norm = (np.max(meas) - np.min(meas))
    elif normalization == "historic":
        norm = np.linalg.norm(meas) * np.sqrt(meas.shape[0])
    elif normalization == "root_of_avg_of_spacedist_squared":  # as in: 2018 Pathak et.al."Hybrid forecasting..."
        norm = np.sqrt(np.mean(np.linalg.norm(meas, axis=1)**2))
    elif utilities._is_number(normalization):
        norm = normalization
    else:
        raise Exception("Type of normalization not implemented")

    return distance/norm


def valid_time_index(error_series, epsilon):
    ''' return the index of the error_series where for the first time error>epsilon
    Args:
        error_series (np.ndarray): array with error for each timestep, shape (T,)
        epsilon (float): Must be equal or greater than 0. The threshhold
    Returns:
        (int): index where error is larger than epsilon
    '''
    if epsilon < 0:
        raise Exception("epsilon must be equal or greater than 0")
    bool_array = error_series > epsilon
    if np.all(bool_array == False):
        return bool_array.size - 1
    else:
        return np.argmax(bool_array)


def demerge_time(pred_time_series, meas_time_series, epsilon):
    """ Synonym for the divergence_time fct. """

    return divergence_time(pred_time_series, meas_time_series, epsilon)


def divergence_time(pred_time_series, meas_time_series, epsilon):
    """ Calculates how long it takes for measurement and prediction to diverge

    Measure for the quality of the predicted trajectory

    The divergence time refers to the number of time_steps it takes for the
    predicted trajectory to diverge from the measured trajectory by more than a
    given distance in one or more dimensions.
    The distance measure is the supremum norm, NOT the euclidean one.

    Args:
        pred_time_series (np.ndarray): predicted/simulated data, shape (T, d)
        meas_time_series (np.ndarray): observed/measured/real data, shape (T, d)
        epsilon (float or np.ndarray): Distance threshold, above which the two
            time series count as diverged. Either float or 1D-array with length d.

    Returns:
        int: divergence_time, the number of time steps for which
            meas_time_series and pred_time_series are separated by less than
            epsilon in each dimension.

    """
    pred = pred_time_series
    meas = meas_time_series

    delta = np.abs(meas - pred)

    div_bool = (delta > epsilon).any(axis=1)
    div_time = np.argmax(np.append(div_bool, True))

    return div_time


def dimension(time_series, r_min=1.5, r_max=5., nr_steps=2,
              plot=False, return_neighbours=False):
    """ Calculates correlation dimension using
    the algorithm by Grassberger and Procaccia.
     
    First we calculate a sum over all points within a given radius, then
    average over all basis points and vary the radius
    (grassberger, procaccia).

    parameters depend on timesteps and the system itself!

    Args:
        time_series (np.ndarray): time series to calculate dimension of, shape (T, d)
        r_min (float): minimum radius
        r_max (float): maximum radius
        nr_steps (int): number of steps in radius, if r_min and r_max are chosen
            properly, then 2 is enough.
        plot (boolean): flag for plotting loglog plot

    Returns: dimension: slope of the log.log plot assumes:
        N_r(radius) ~ radius**dimension
    """

    nr_points = float(time_series.shape[0])
    radii = np.logspace(np.log10(r_min), np.log10(r_max), nr_steps)

    tree = scipy.spatial.cKDTree(time_series)
    N_r = np.array(tree.count_neighbors(tree, radii), dtype=float) / nr_points
    N_r = np.vstack((radii, N_r))

    if nr_steps > 2:
        # linear fit based on loglog scale, to get slope/dimension:
        slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)[0:2]
        dimension = slope
    elif nr_steps == 2:
        slope = (np.log(N_r[1, 1]) - np.log(N_r[1, 0])) / (np.log(N_r[0, 1]) - np.log(N_r[0, 0]))
        dimension = slope

    ###plotting
    # if plot:
    #     plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.)
    #     plt.title('loglog plot of the N_r(radius), slope/dim = ' + str(slope))
    #     plt.show()
    if plot:
        warn_string = "Plotting was removed in the entirety of the rescomp package.\n" \
                      "The 'plot' paramter will be removed in future releases as well."
        warnings.warn(warn_string, UserWarning)

    if return_neighbours:
        return dimension, N_r
    else:
        return dimension


def dimension_parameters(time_series, nr_steps=100, literature_value=None,
                         plot=False, r_minmin=None, r_maxmax=None,
                         shortness_weight=0.5, literature_weight=1.):
    """ Estimates parameters r_min and r_max for calculation of correlation
    dimension using the algorithm by Grassberger and Procaccia and uses them 
    to calculate it.
     
    This experimental function performs a simple grid search on r_min and r_max
    in the intervall given by r_minmin, r_maxmax and nr_steps. The performance 
    of the parameters is measured by a combination of NRMSE, a penalty for small 
    intervalls relative to given r_minmin and r_maxmax and a quadratic penalty 
    for the difference from the literature value if given.
    
    For calculating the dimension of a high number of similar time_series in a 
    row it is advisable to use this function only once to get the parameters 
    and then use the function dimension with them in the subsequent computations. 
    
    Might fail for short time_series or unreasonable choices of parameters. 
    It is recommended to use the plot option to double check the plausibility 
    of the results.

    Args:
        time_series (np.ndarray): time series to calculate dimension of, shape (T, d)
        r_minmin (float): minimum radius in grid search
        r_maxmax (float): maximum radius in grid search
        nr_steps (int): number of steps in grid search
        plot (boolean): flag for plotting loglog plot

     Returns:
            tuple: 3-element tuple containing:

            - **best_r_min** (*float*): Estimation for r_min
            - **best_r_max** (*float*): Estimation for r_max
            - **dimension** (*float*): Estimation for dimension using 
              the parameters best_r_min and best_r_max
    """

    if r_maxmax is None:
        expansion = []
        for d in range(time_series.shape[1]):
            expansion.append(np.max(time_series[:, d] - np.min(time_series[:, d])))

        r_maxmax = np.max(expansion)

    if r_minmin is None:
        r_minmin = 0.001 * r_maxmax

    literature_cost = 0

    nr_points = float(time_series.shape[0])
    radii = np.logspace(np.log10(r_minmin), np.log10(r_maxmax), nr_steps)

    tree = scipy.spatial.cKDTree(time_series)
    N_r = np.array(tree.count_neighbors(tree, radii), dtype=float) / nr_points
    N_r = np.vstack((radii, N_r))

    loss = None

    for start_index in range(nr_steps - 1):
        for end_index in range(start_index + 1, nr_steps):
            # print(str(start_index)+', '+ str(end_index))
            current_N_r = N_r[:, start_index:end_index]
            current_r_min = radii[start_index]
            current_r_max = radii[end_index]

            # linear fit based on loglog scale, to get slope/dimension:
            slope, intercept = np.polyfit(np.log(current_N_r[0]),
                                          np.log(current_N_r[1]), deg=1)[0:2]

            dimension = slope

            estimated_line = intercept + slope * np.log(current_N_r[0])
            error = rmse(np.log(current_N_r[1]), estimated_line,
                         normalization="historic")
            shortness_cost = nr_steps / (end_index - start_index) ** 3

            if literature_value is not None:
                literature_cost = np.sqrt(literature_value - dimension)

            new_loss = error + shortness_weight * shortness_cost + literature_weight * literature_cost * 5.

            if loss is None:

                loss = new_loss
                best_r_min = current_r_min
                best_r_max = current_r_max

                best_slope = slope
                best_intercept = intercept

            elif new_loss < loss:
                loss = new_loss

                best_r_min = current_r_min
                best_r_max = current_r_max

                best_slope = slope
                best_intercept = intercept

    dimension = best_slope

    # ###plotting
    # if plot:
    #
    #     plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.,label='data')
    #     plt.loglog(N_r[0], best_intercept + best_slope*N_r[1],
    #              label='fit: r_min ='+str(round(best_r_min,3))+', r_max = '+
    #              str(round(best_r_max,3)))
    #     plt.axvline(x=best_r_min)
    #     plt.axvline(x=best_r_max)
    #     plt.title('loglog plot of the N_r(radius), slope/dim = ' + str(dimension))
    #     plt.legend()
    #     plt.show()
    if plot:
        warn_string = "Plotting was removed in the entirety of the rescomp package.\n" \
                      "The 'plot' paramter will be removed in future releases as well."
        warnings.warn(warn_string, UserWarning)

    return best_r_min, best_r_max, dimension


def iterator_based_lyapunov_spectrum(f, starting_points, T=1, tau=0, eps=1e-6, nr_of_lyapunovs=None,
                                     nr_steps=3000, dt=1.,return_convergence=False, return_traj_divergence=False,
                                     jacobian=None, agg=None):
    '''
    The Algorithm is based on: 1902.09651 "LYAPUNOV EXPONENTS of the KURAMOTO-SIVASHINSKY PDE"
    For its explanation see: 0811.0882 "The lyapunov characteristic exponents and their computation"

    Calculates the lyapunov spectrum of a discrete dynamical system
    with x_(n+1) = f(x_n) using a standard QR-based algorithm, where the timeevolution of the deviation vectors is
    calculated by actually simulating the trajectories and the deviation vectors

    Based on the iterator function f. The Jacobian is calculated numerically or if explicitly given analytically

    Measure for chaotic behaviour in the system.

    Important characteristic to compare attractors.

    Args:
        f (function): mapping with x_n+1 = f(x_n)
        starting_points (np.ndarray): inintial condition of iteration:
            possbilities:
            - np.ndarray with shape = (state_dim) -> use as initial condition for iterations
            - np.ndarray with shape = (N_ens, state_dim) -> do the whole calculation for N_ens starting points
            - None -> parse None to the iteration function f(None) -> f must have a default for None
        T (float): time interval between successive QR decompositions
        tau (float): time to simulate system before exponent computation
        eps (float): perturbation magnitude in space, to approximate the jacobian
        nr_of_lyapunovs (int): nr of greatest lyapunov components you which to calculate
                               if None: all lyapunov exponents are calculated
        nr_steps (int): The number of reorthonormalisation steps
        dt (float): If the iterator corresponds to a contnuous system -> the time between
                    two succesive steps
        return_convergence (bool): If true, additionally to the lyapunov exponents, return
                                   the convergence according to the N steps
       return_traj_divergence (bool): If true, also return the divergence of the trajectories within every reorthostep
                                    Only if jacobian is None
        jacobian(None or function):
            - If None: The jacobian is calculated numerically using the distance "eps"
            - If Func: The jacobian is passed as a function that takes the point x as input
                       and outputs the jacobian at this point
        agg(bool, str or list): only has effect if a ensemble is calculated:
            - None -> return all ensemble LEs
            - mean -> return mean of ensemble LEs
            - std -> return std of ensemble LEs
            - combination of above in list -> return all in list
    Returns: lyapunov spectrum if return_convergence is False,
                                 tuple of final lyapunov spectrum and development
                                 of lyapunov spectrum if return_convergence is
                                 True
    '''
    if (return_traj_divergence) and (jacobian is not None):
        raise Exception("traj divergence can not be computed (yet) when a jacobian is given") # TODO implement

    def f_steps(x, steps, save_all=False):
        if save_all:
            out = np.zeros((steps+1, x.size))
            out[0, :] = x

        for i in range(steps):
            x = f(x)
            if save_all:
                out[i+1, :] = x
        if save_all:
            return out
        else:
            return x

    m = nr_of_lyapunovs
    N = nr_steps

    # handling the time steps
    tau_timesteps = int(tau/dt)
    T_timesteps = int(T/dt)
    tau_new = tau_timesteps*dt
    T_new = T_timesteps*dt
    if tau != tau_new:
        print(f"Updated tau to multiple of dt: tau = {tau}")
        tau = tau_new
    if T != T_new:
        print(f"Updated T to multiple of dt: T = {T}")
        T = T_new

    # handling the starting point
    if starting_points is not None:
        if len(starting_points.shape) == 2:
            N_ens = starting_points.shape[0]
        else:
            starting_points = starting_points[np.newaxis, :]
            N_ens = 1
        state_dim = starting_points[0, :].size
    elif starting_points is None:
        y = f(starting_points, 1)
        if y is None:
            raise Exception("iterator f does not support None as input")
        state_dim = y.size
        N_ens = 1
    if m is None: # calculate whole spectrum of lyapunov exponents
        m = state_dim
    else:
        if m > state_dim:
            raise Exception(f"required number of lyapunov exponents, larger than state-dimension: {m} vs. {state_dim}")

    lyapunov_exp_ens = np.zeros((N_ens, m))
    if return_convergence:
        lyapunov_exp_convergence_ens = np.zeros((N_ens, N, m))

    if return_traj_divergence:
        traj_divergence_ens = np.zeros((N_ens, N, m, T_timesteps+1))

    for i_ens in range(N_ens):
        if N_ens > 1:
            print(f"N_ens: {i_ens + 1}/{N_ens}")
        if starting_points is not None:
            starting_point = starting_points[i_ens, :]

        if tau_timesteps == 0:
            x = starting_point
        else:
            x = f_steps(starting_point, tau_timesteps)  # discard transient states

        # choose m initial orthonormal directions:
        Q = np.eye(state_dim, m)

        # initialize the matrix W, that holds the deviation vectors:
        W = np.zeros(Q.shape)

        # Initialize a Matrix to store the R_diags:
        R_diags = np.zeros((N, m))

        for j in range(1, N + 1):  # "for all time intervals"
            if jacobian is not None:
                x_new = x # every x is needed to evolve the deviation vectors
                for i in range(T_timesteps):
                    local_jac = jacobian(x_new)
                    x_new = f(x_new)
                    def Q_it(y):
                        return local_jac.dot(y)
                    Q = simulations._runge_kutta(Q_it, dt, Q) # alternatively: Q = Q + local_jac.dot(Q)*dt
                W = Q
            else:
                # x_new = f_steps(x, T_timesteps)
                # for i in range(m):  # for all m orthonormal directions
                #     q_i = Q[:, i]
                #     x_mod_i = x + eps*q_i
                #     x_mod_i_new = f_steps(x_mod_i, T_timesteps)
                #     W[:, i] = (x_mod_i_new - x_new)/eps

                # alt:
                if return_traj_divergence:
                    x_new_traj = f_steps(x, T_timesteps, save_all=True)
                    x_new = x_new_traj[-1, :]
                else:
                    x_new = f_steps(x, T_timesteps) # if T_timesteps = 1 -> it iterates the function one time
                for i in range(m):  # for all m orthonormal directions
                    q_i = Q[:, i]
                    x_mod_i = x + eps*q_i
                    if return_traj_divergence:
                        x_mod_i_new_traj = f_steps(x_mod_i, T_timesteps, save_all=True)
                        x_mod_i_new = x_mod_i_new_traj[-1, :]

                        div_i = np.linalg.norm((x_mod_i_new_traj - x_new_traj), axis=-1) # calculate the distance
                        traj_divergence_ens[i_ens, j-1, i, :] = div_i # /eps
                    else:
                        x_mod_i_new = f_steps(x_mod_i, T_timesteps)
                    W[:, i] = (x_mod_i_new - x_new)/eps
                # end alt

            Q, R = np.linalg.qr(W)
            for i in range(m):
                R_diags[j-1, i] = R[i, i]
            x = x_new

        lyapunov_exp = np.sum(np.log(np.abs(R_diags))/(N*T), axis = 0)

        lyapunov_exp_ens[i_ens, :] = lyapunov_exp

        if return_convergence:
            times = np.arange(1, N+1)*T
            lyapunov_exp_convergence = np.cumsum(np.log(np.abs(R_diags)), axis = 0)/(np.tile(times, (m, 1)).T)
            lyapunov_exp_convergence_ens[i_ens, :, :] = lyapunov_exp_convergence

    if not type(agg) in (list, tuple):
        agg = [agg, ]

    to_return = []
    if return_convergence:
        to_return_conv = []
    if return_traj_divergence:
        to_return_traj_div = []
    for a in agg:
        if a is None:
            to_return.append(lyapunov_exp_ens)
            if return_convergence:
                to_return_conv.append(lyapunov_exp_convergence_ens)
            if return_traj_divergence:
                to_return_traj_div.append(traj_divergence_ens)
        elif a == "mean":
            to_return.append(np.mean(lyapunov_exp_ens, axis=0))
            if return_convergence:
                to_return_conv.append(np.mean(lyapunov_exp_convergence_ens, axis=0))
            if return_traj_divergence:
                to_return_traj_div.append(np.mean(traj_divergence_ens, axis=0))
        elif a == "std":
            to_return.append(np.std(lyapunov_exp_ens, axis=0))
            if return_convergence:
                to_return_conv.append(np.std(lyapunov_exp_convergence_ens, axis=0))
            if return_traj_divergence:
                to_return_traj_div.append(np.std(traj_divergence_ens, axis=0))

    master_return = []

    if len(to_return) == 1:
        to_return = to_return[0]
        master_return.append(to_return)
    if return_convergence:
        if len(to_return_conv) == 1:
            to_return_conv = to_return_conv[0]
        master_return.append(to_return_conv)
    if return_traj_divergence:
        if len(to_return_traj_div) == 1:
            to_return_traj_div = to_return_traj_div[0]
        master_return.append(to_return_traj_div)

    if len(master_return) == 1:
        master_return = master_return[0]

    return master_return

def KY_dimension(lyapunov_exponents):
    '''
    Calculates the Kaplan-Yorke dimension from the lyapunov_exponents spectrum.
    This requires that the sum of all lyapunov exponents is negative.
    Following: "LYAPUNOV EXPONENTS of the KURAMOTO-SIVASHINSKY PDE" (1902.09651)
    '''
    # sort just to make sure:
    lyapunov_sorted = np.sort(lyapunov_exponents)[::-1] # ascending order
    cumsum = lyapunov_sorted.cumsum()
    j = np.where((cumsum>=0))[0].max() + 1 # +1 since index starts from 0
    if j == len(lyapunov_exponents):
        raise Exception("Lyapunov Exponents are \"too positive\" to calculate the Kaplan-Yorke dimension")
    D_KY = j + cumsum[j-1]/np.abs(lyapunov_sorted[j])
    return D_KY


## simple LE Algorithm:
def calculate_divergence(f, starting_points, T=1, tau=0, dt=1., eps=1e-6, N_dims=1, agg=None, random_directions=False):
    '''
    TODO: Clear up, add reference etc.
    Args:
        f:
        starting_points:
        T:
        tau:
        dt:
        eps:
        N_dims:
    Returns:
    '''
    def f_steps(x, steps):
        for i in range(steps):
            x = f(x)
        return x

    # handling the time steps
    tau_timesteps = int(tau/dt)
    T_timesteps = int(T/dt)
    tau_new = tau_timesteps*dt
    T_new = T_timesteps*dt
    if tau != tau_new:
        print(f"Updated tau to multiple of dt: tau = {tau}")
        tau = tau_new
    if T != T_new:
        print(f"Updated T to multiple of dt: T = {T}")
        T = T_new

    # handling the starting point
    if starting_points is not None:
        if len(starting_points.shape) == 2:
            N_ens = starting_points.shape[0]
        else:
            starting_points = starting_points[np.newaxis, :]
            N_ens = 1
        state_dim = starting_points[0, :].size
    else:
        state_dim = f(starting_points, 1).size
        N_ens = 1

    if not random_directions:
        if N_dims is None:
            N_dims = state_dim
        else:
            if N_dims > state_dim:
                raise Exception(f"N_dims larger than state-dimension: {N_dims} vs. {state_dim}")

    deviation_trajectory_ens = np.zeros((T_timesteps + 1, state_dim, N_dims, N_ens))
    for i_ens in range(N_ens):
        print(f"N_ens: {i_ens + 1}/{N_ens}")
        if starting_points is not None:
            starting_point = starting_points[i_ens, :]
        if tau_timesteps == 0:
            x = starting_point
        else:
            print("..calculating transient..")
            x = f_steps(starting_point, tau_timesteps)  # discard transient states

        if random_directions:
            initial_deviations = np.random.randn(state_dim, N_dims) * eps
        else:
            initial_deviations = np.eye(state_dim, N_dims) * eps

        basis_trajectory = np.zeros((T_timesteps + 1, state_dim))
        basis_trajectory[0, :] = x

        perturbed_trajectory = np.zeros((T_timesteps + 1, state_dim, N_dims))
        for i in range(N_dims):
            perturbed_trajectory[0, :, i] = x + initial_deviations[:, i]

        for i_t in range(1, T_timesteps + 1):
            if (i_t) % 10 == 0:
                print(f"timestep {i_t}/{T_timesteps}", end="\r")
            x = f(x)
            basis_trajectory[i_t, :] = x

            for i in range(N_dims):
                perturbed_trajectory[i_t, :, i] = f(perturbed_trajectory[i_t - 1, :, i])
        print("")
        deviation_trajectory = perturbed_trajectory - np.repeat(basis_trajectory[:, :, np.newaxis], N_dims, axis=2)
        deviation_trajectory_ens[:, :, :, i_ens] = deviation_trajectory

    deviation_len_traj_ens = np.linalg.norm(deviation_trajectory_ens, axis=1)

    # aggregation
    if not type(agg) in (list, tuple):
        agg = [agg, ]
    to_return = []

    for a in agg:
        if a is None:
            to_return.append(deviation_len_traj_ens)

        elif a == "mean":
            to_return.append(np.mean(deviation_len_traj_ens, axis=(-1, -2)))

        elif a == "std":
            to_return.append(np.std(deviation_len_traj_ens, axis=(-1, -2)))

    if len(to_return) == 1:
        to_return = to_return[0]

    return to_return


def fourier_spectrum(time_series, period=False, dt=1):
    """
    Calculates the fourier spectrum of the n-dimensional time_series.
    For every dimension, calculate the FFT and then take the L2 norm over the dimensions.
    Return the results over the preiod (time-domain) or frequency (frequency-domain)
    Args:
        time_series (np.ndarray): time series to transform, shape (T, d)
        period (bool): if True return time as xout
        dt: if the timeincrement of the time_series is known

    Returns:
        xout (np.ndarray): the x axis
        yout (np.ndarray): the fft of the signal (with norm over dims)
    """
    # fourier transform:
    fourier = np.fft.fftn(time_series)
    mean_fourier = np.linalg.norm(fourier, axis=-1)

    freq = np.fft.fftfreq(time_series[:, 0].size)

    N = mean_fourier.size
    half_fourier = mean_fourier[1:int(N/2)]
    half_freq = freq[1:int(N/2)]/dt

    yout = half_fourier
    if period:
        half_period = 1/half_freq
        xout = half_period
    else:
        xout = half_freq

    return xout, yout


def lyapunov_rosenstein(time_series, dt=1.0, freq_cut=True, pnts_to_try=50, steps=100, verb=1,
                        debug=False):
    """
    A variation of the rosenstein algorithm to extract the lyapunov exponent from a time_series. Embedding is not
    implemented.
    Original Paper: Rosenstein et. al. (1992)
    https://doi.org/10.1016/0167-2789(93)90009-P

    Returns the mean logarithmic distance between close trajectories and the corresponding x axis
    -> by fitting the linear region of this curve, the largest lyapunov exponent can be obtained by the sloap.

    Args:
        time_series (np.ndarray): the time series, shape (T, d)
        dt (float): the time step of the time series
        freq_cut (bool): If true, only consider neighbouring points that are at least the mean period (temporaly) apart
        pnts_to_try (int): If freq_cut is True -> Nr of nearest neighbours to try, to check if the they are at least +
                           the mean period apart
        steps (int): The nr of steps to follow the base and neighbour point
        verb (int): If 1: Print out more info
        debug(bool): If true-> return more quantities that might be useful for debugging

    Returns:
        By fitting the linear region in t_list vs. avg_log_dist, the sloap is the largest Lyapunov Exponent
        if debug False:
            return (avg_log_dist, t_list)
        if debug True:
            if freq_cut True
                return (avg_log_dist, t_list, index_distance_array, d_no_zero, time, amplitude, avg_period)
            if freq_cut False
                return (avg_log_dist, t_list, index_distance_array, d_no_zero)
    """

    if freq_cut:
        time, amplitude = fourier_spectrum(time_series, period=True, dt=dt)
        avg_period = np.sum(time * amplitude) / np.sum(amplitude)
        if verb == 1:
            print(f"avg period: {avg_period}")

    tree = scipy.spatial.cKDTree(time_series)
    nr_points = time_series.shape[0]
    neighbour_array = np.zeros((nr_points,), dtype=float)
    if debug:
        index_distance_array = np.zeros((nr_points,), dtype=float)

    if freq_cut:
        period_cut = avg_period
        for it in range(nr_points):
            x = time_series[it, :]
            query = tree.query(x, k=pnts_to_try)
            potential_neighbours = query[1][1:]
            index_distance = np.abs(potential_neighbours - it)
            larger_than_period_cut = index_distance > avg_period
            neighs_larger_than_period_cut = potential_neighbours[larger_than_period_cut]
            if len(neighs_larger_than_period_cut) > 0:
                if debug:
                    index_distance_array[it] = index_distance[larger_than_period_cut][0]
                neighbour = neighs_larger_than_period_cut[0]
                neighbour_array[it] = neighbour
            else:
                if debug:
                    index_distance_array[it] = np.NaN
                neighbour_array[it] = np.NaN
        if verb == 1:
            nans = np.isnan(neighbour_array)
            print(
                f"For {nans.sum()}/{nans.size} points, all {pnts_to_try} closest neighbours were temporally closer than {period_cut} and thus not considered")
    else:
        for it in range(nr_points):
            x = time_series[it, :]
            neighbour = tree.query(x, k=2)[1][1]
            neighbour_array[it] = neighbour
            if debug:
                index_distance_array[it] = np.abs(neighbour - it)

    distance_array = np.empty((nr_points, steps))
    distance_array[:, :] = np.NaN

    if verb == 1:
        nans_pre = nans.sum()

    for i_base, i_neigh in enumerate(neighbour_array):
        if np.isnan(i_neigh):
            continue
        else:
            i_neigh = int(i_neigh)
            if i_base + steps < nr_points and i_neigh + steps < nr_points:
                diff = time_series[i_base:i_base + steps, :] - time_series[i_neigh:i_neigh + steps, :]
                distance_array[i_base, :] = np.linalg.norm(diff, axis=-1)

    if verb == 1:
        nans_2 = np.isnan(distance_array).any(axis=1)
        print(f"For {nans_2.sum() - nans_pre}/{nans_2.size - nans_pre} points, "
              f"there were not {steps} steps left in the timeseries (either for the base and/or nn-point)")

    # remove rows with nan
    d = distance_array[~np.isnan(distance_array).any(axis=1)]
    # remove 0 distance:
    d_no_zero = d[(d != 0).any(axis=1)]

    if verb == 1:
        final_nr_of_pnts = d_no_zero.shape[0]
        print(f"final number of points: {final_nr_of_pnts}")

    log_d = np.log(d_no_zero)

    avg_log_dist = np.mean(log_d, axis=0)
    t_list = np.arange(steps)*dt

    if debug:
        if freq_cut:
            to_return = (avg_log_dist, t_list, index_distance_array, d_no_zero, time, amplitude, avg_period)
        else:
            to_return = (avg_log_dist, t_list, index_distance_array, d_no_zero)
    else:
        to_return = (avg_log_dist, t_list)

    return to_return


def poincare_map(time_series, mode="minima", dimension=0):
    x = time_series[:, dimension]
    if mode == "minima":
        ix = argrelextrema(x, np.less)[0]
    elif mode == "maxima":
        ix = argrelextrema(x, np.greater)[0]
    else:
        raise Exception(f"mode: {mode} not recognized")
    extreme = x[ix]

    return extreme[:-1], extreme[1:]

pass  # TODO: Generalize Joschka's Lyap. Exp. Sprectrum from Reservoir code
# def reservoir_lyapunov_spectrum(esn, nr_steps=2500, return_convergence=False,
#                                 dt=1., starting_point=None):
#     """ Calculates the lyapunov spectrum of the esn using a standard QR-based
#     algorithm.
#
#    Calls equation_based_lyapunov_spectrum_discrete()
#
#     Args:
#         f (function): mapping with x_n+1 = f(x_n)
#         Jacobian (function): Jacobian of f , takes x as argument
#         starting_point (np.ndarray): inintial condition of iteration
#         nr_steps (int): number of iteration steps
#         dt (float): time scale of a step
#         return_convergence (bool): if True returns the development of the
#                 estimate for the lyapunov spectrum over time in steps of 100
#                 iterations.
#
#     Returns:
#         np.ndarray_or_tuple : lyapunov spectrum if return_convergence is False,
#                                 tuple of final lyapunov spectrum and development
#                                 of lyapunov spectrum if return_convergence is
#                                 True
#
#
#      """
#
#     d = esn._network.shape[0]
#     if esn._act_fct_flag == 0 and esn._w_out_fit_flag == 0:
#         # Standard tanh activation function, linear readout, no bias
#
#         M = np.array(esn._w_in @ esn._w_out + esn._network)
#
#         def f(r):
#             return np.tanh(M @ r)
#
#         # def Jacobian1(r):
#         #    M_r=M@r
#         #    J=np.zeros((d,d))
#         #
#         #    for i in range(d):
#         #        for j in range(d):
#         #            J[i,j] = M[i,j]/(np.cosh(M_r[i])**2)
#         #    return np.array(J)
#
#         def Jacobian(r):
#             '''
#             M_r=M@r
#
#             J = np.cosh(M_r)**(-2)*M
#
#             return np.array(J)
#             '''
#             M_r = M @ r
#
#             J = (np.cosh(M_r) ** (-2) * M.T).T
#
#             # J[np.abs(J) <= 0.001 * np.max(np.abs(J))] = 0
#             # J_sparse = scipy.sparse.csr_matrix(J)
#             # return np.array(J)
#             return J
#
#     elif esn._act_fct_flag == 0 and esn._w_out_fit_flag == 1:
#         # Standard tanh activation function, linear and squared readout, no bias
#
#         W_out_1 = esn._w_out[:, :d]
#         W_out_2 = esn._w_out[:, d:]
#
#         # M  = np.array(esn._w_in@esn._w_out + esn._network)
#
#         def f(r):
#             return np.tanh(
#                 esn._network @ r + esn._w_in @ (W_out_1 @ r + W_out_2 @ r ** 2))
#
#         # def Jacobian1(r):
#         #    M_r=esn._network@r + esn._w_in@(W_out_1@r + W_out_2@r**2)
#         #
#         #    J=np.zeros((d,d))
#         #
#         #    M_1 = esn._network + esn._w_in@W_out_1
#         #    M_2 = esn._w_in@W_out_2
#         #
#         #    for i in range(d):
#         #        for j in range(d):
#         #            J[i,j] = (M_1[i,j]+2*M_2[i,j]*r[j])/(np.cosh(M_r[i])**2)
#         #    return np.array(J)
#
#         def Jacobian(r):
#             '''
#             #B = 2*esn._w_in@W_out_2*r
#             M_r=esn._network@r + esn._w_in@(W_out_1@r + W_out_2@r**2)
#
#             J = np.cosh(M_r)**(-2)*(esn._network + esn._w_in@W_out_1 +
#                                     2*esn._w_in@W_out_2*r)
#             '''
#
#             M_r = esn._network @ r + esn._w_in @ (
#                         W_out_1 @ r + W_out_2 @ (r ** 2))
#             J = ((esn._network.T + (esn._w_in @ W_out_1).T +
#                   2 * (esn._w_in @ W_out_2 * r).T) / (np.cosh(M_r) ** 2)).T
#
#             return np.array(J)
#
#     elif esn._act_fct_flag == 0 and esn._w_out_fit_flag == 2:
#         # Standard tanh activation function, linear readout with bias
#
#         W_out = esn._w_out[:, :-1]
#
#         b = esn._w_out[:, -1:].reshape(esn._w_out.shape[0], )
#
#         M = np.array(esn._w_in @ W_out + esn._network)
#
#         def f(r):
#             return np.tanh(M @ r + esn._w_in @ b)
#
#         # def Jacobian1(r):
#         #    M_r=M@r+esn._w_in@b
#         #
#         #    J=np.zeros((d,d))
#         #
#         #    for i in range(d):
#         #        for j in range(d):
#         #            J[i,j] = M[i,j]/(np.cosh(M_r[i])**2)
#         #
#         #    return np.array(J)
#
#         def Jacobian(r):
#             M_r = M @ r + esn._w_in @ b
#
#             J = (np.cosh(M_r) ** (-2) * M.T).T
#
#             return np.array(J)
#
#     elif esn._act_fct_flag == 1 and esn._w_out_fit_flag == 0:
#         # Tanh activation function with bias, linear readout
#
#         M = np.array(esn._w_in @ esn._w_out + esn._network)
#
#         bias = esn._bias
#
#         def f(r):
#             return np.tanh(M @ r + bias)
#
#         # def Jacobian1(r):
#         #    M_r=M@r
#         #    J=np.zeros((d,d))
#         #
#         #    for i in range(d):
#         #        for j in range(d):
#         #            J[i,j] = M[i,j]/(np.cosh(M_r[i] + bias[i])**2)
#         #    return np.array(J)
#
#         def Jacobian(r):
#             '''
#             M_r=M@r
#
#             J = np.cosh(M_r + bias)**(-2)*M
#
#             return np.array(J)
#             '''
#             M_r = M @ r
#
#             J = (np.cosh(M_r + bias) ** (-2) * M.T).T
#
#             return np.array(J)
#
#     elif esn._act_fct_flag == 3 and esn._w_out_fit_flag == 0:
#         # Mix of normal tanh and tanh^2 activation functions, linear readout,
#         # no bias
#
#         M = np.array(esn._w_in @ esn._w_out + esn._network)
#
#         def f(r):
#             new_r = np.zeros(r.shape)
#             new_r[esn._normal_tanh_nodes] = np.tanh(M @ r)[
#                 esn._normal_tanh_nodes]
#             new_r[esn._squared_tanh_nodes] = np.tanh(M @ r)[
#                                                  esn._squared_tanh_nodes] ** 2
#             return new_r
#
#         # def Jacobian1(r):
#         #    M_r=M@r
#         #    J=np.zeros((d,d))
#         #
#         #    for i in range(d):
#         #        for j in range(d):
#         #            if i in esn._normal_tanh_nodes:
#         #                J[i,j] = M[i,j]/(np.cosh(M_r[i])**2)
#         #            elif i in esn._squared_tanh_nodes:
#         #                J[i,j] = 2*M[i,j]*np.tanh(M_r[i])/(np.cosh(M_r[i])**2)
#         #    return np.array(J)
#
#         def Jacobian(r):
#
#             M_r = M @ r
#
#             J = np.zeros((r.shape[0], r.shape[0]))
#
#             J[esn._normal_tanh_nodes] = (
#                         np.cosh(M_r[esn._normal_tanh_nodes]) ** (-2) *
#                         M[esn._normal_tanh_nodes].T).T
#             J[esn._squared_tanh_nodes] = (
#                         2 * np.tanh(M_r[esn._squared_tanh_nodes]) *
#                         np.cosh(M_r[esn._squared_tanh_nodes]) ** (-2) *
#                         M[esn._squared_tanh_nodes].T).T
#
#             return np.array(J)
#
#     else:
#         raise Exception(
#             "reservoir_lyapunov_spectrum not implemented for this " +
#             "activation function and readout")
#
#     if starting_point is None:
#         starting_point = esn._last_r
#
#     return equation_based_lyapunov_spectrum_discrete(f, Jacobian,
#                                                      starting_point=starting_point,
#                                                      nr_steps=nr_steps, dt=dt,
#                                                      return_convergence=return_convergence)
#
#
# def equation_based_lyapunov_spectrum_discrete(f, Jacobian, starting_point,
#                                               nr_steps=3000, dt=1.,
#                                               return_convergence=False):
#     """ Calculates the lyapunov spectrum of a discrete dynamical system
#     with x_n+1 = f(x_n) using a standard QR-based algorithm.
#
#     Based on equations not data.
#
#     Measure for chaotic behaviour in the system.
#
#     Important characteristic to compare attractors.
#
#     Args:
#         f (function): mapping with x_n+1 = f(x_n)
#         Jacobian (function): Jacobian of f , takes x as argument
#         starting_point (np.ndarray): inintial condition of iteration
#         nr_steps (int): number of iteration steps
#         dt (float): time scale of a step
#         return_convergence (bool): if True returns the development of the
#                 estimate for the lyapunov spectrum over time in steps of 100
#                 iterations.
#
#     Returns:
#         np.ndarray_or_tuple : lyapunov spectrum if return_convergence is False,
#                                 tuple of final lyapunov spectrum and development
#                                 of lyapunov spectrum if return_convergence is
#                                 True
#
#
#      """
#     x = starting_point
#
#     # jac = Jacobian(x)
#     # jac_sparse = scipy.sparse.csr_matrix(Jacobian(x))
#
#     Q, R = np.linalg.qr(Jacobian(x))
#
#     s = []
#     lces = []
#
#     for n in range(nr_steps):
#         x = f(x)
#         Q, R = np.linalg.qr(np.matmul(Jacobian(x), Q))
#
#         s.append(np.array([R[i, i] for i in range(len(R))]))
#
#         # if n % 100 == 0 and return_convergence:
#         #     lya = np.sum(np.log(np.abs(s)), axis=0) / (n * dt)
#         #     lces.append(lya)
#
#         if return_convergence:
#             lya = np.sum(np.log(np.abs(s)), axis=0) / ((n + 1) * dt)
#             lces.append(lya)
#
#     lya = np.sum(np.log(np.abs(s)), axis=0) / (nr_steps * dt)
#
#     if return_convergence:
#         return lya, np.array(lces)
#     else:
#         return lya


pass  # Either generalize or remove the old functions from Jonas
# def lyapunov_from_data(traj, dt, threshold=int(10),
#                        plot=False):
#     """
#     Calculates the maximal Lyapunov Exponent of reservoir.y_pred (or reservoir.y_test),
#     by estimating the time derivative of the mean logarithmic distances of
#     former next neighbours. Stores it in reservoir.lyapunov (reservoir.lyapunov_test)
#     Only values for tau_min/max are used for calculating the slope!
#
#     Since the attractor has a size of roughly 20 [log(20) ~ 3.] this
#     distance reaches a maximum after a certain time, approximately
#     after 4. time units [time_units = dt*steps]
#     Therefore the default values are choosen to be dt dependent as in
#     ###Definition of taus:
#
#     tau_min/max are given in units of steps
#     plot to check for correct average
#     """
#     """
#     REMINDER:
#     remove the loop over taus, since the slope is calculated with single
#     values only
#     """
#     ###Definition of taus:
#     tau_min = int(0.5 / dt)
#     tau_max = int(3.8 / dt)
#     taus = np.arange(tau_min, tau_max, 10)
#     # taus = np.array([tau_min, tau_max])
#
#     tree = scipy.spatial.cKDTree(traj)
#     nn_index = tree.query(traj, k=2)[1]
#
#     # drop all elements in nn_index lists where the neighbour is:
#     # 1. less than threshold time_steps away
#     # 2. where we cannot calculate the neighbours future in tau_max time_steps:
#
#     # contains indices of points and the indices of their nn:
#
#     nn_index = nn_index[nn_index[:, 1] + tau_max < traj.shape[0]]
#     nn_index = nn_index[nn_index[:, 0] + tau_max < traj.shape[0]]
#
#     # Calculate the largest Lyapunov exponent:
#     # for storing the results:
#     Sum = []
#     # loop over differnt tau, to get a functional dependence:
#     for tau in taus:
#         # print(tau)
#
#         S = []  # the summed values for all basis points
#
#         # loop over every point in the trajectory, where we can calclutate
#         # the future in tau_max time_steps:
#         for point, nn in nn_index:
#             S.append(np.log(np.linalg.norm(traj[point + tau] - traj[
#                 nn + tau])))  # add one points average s to S
#
#             # since there is no else, we only avg over the points that have
#             # points in their epsilon environment
#         Sum.append((tau * dt, np.array(S).mean()))
#     Sum = np.array(Sum)
#
#     slope = (Sum[-1, 1] - Sum[0, 1]) / (Sum[-1, 0] - Sum[0, 0])
#     if plot:
#         plt.title('slope: ' + str(slope))
#         plt.plot(Sum[:, 0], Sum[:, 1])
#         plt.plot(Sum[:, 0], Sum[:, 0] * slope)
#         plt.xlabel('time dt*tau[steps]')
#         plt.ylabel('log_dist_former_neighbours')
#         # plt.plot(Sum[:,0], Sum[:,0]*slope + Sum[0,0])
#         plt.show()
#
#     return slope

# def return_map(self, axis=2):
#     """
#     Shows the recurrence plot of the maxima of a given axis
#     """
#     max_pred = self.y_pred[scipy.signal.argrelextrema(self.y_pred[:,2],
#         np.greater, order = 5),axis]
#     max_test = self.y_test[scipy.signal.argrelextrema(self.y_test[:,2],
#         np.greater, order=5),axis]
#     plt.plot(max_pred[0,:-1:2], max_pred[0,1::2],
#              '.', color='red', alpha=0.5, label='predicted y')
#     plt.plot(max_test[0,:-1:2], max_test[0,1::2],
#              '.', color='green', alpha=0.5, label='test y')
#     plt.legend(loc=2, fontsize=10)
#     plt.show()


# def dimension(reservoir, r_min=0.5, r_max=5., r_steps=0.15,
#               plot=False, test_measure=False):
#     """ Calculates correlation dimension
#
#     for reservoir.y_pred (or reservoir.y_test) using
#     the algorithm by Grassberger and Procaccia and returns dimension.
#     traj: trajectory of an attractor, whos correlation dimension is returned
#     First we calculate a sum over all points within a given radius, then
#     average over all basis points and vary the radius
#     (grassberger, procaccia).
#
#     parameters depend on reservoir.dt and the system itself!
#
#     N_r: list of tuples: (radius, average number of neighbours within all
#         balls)
#
#     Args:
#         reservoir ():
#         r_min ():
#         r_max ():
#         r_steps ():
#         plot ():
#         test_measure ():
#
#     Returns: dimension: slope of the log.log plot assumes:
#         N_r(radius) ~ radius**dimension
#
#     """
#     if test_measure:
#         traj = reservoir.y_test  # for measure assessing
#     else:
#         traj = reservoir.y_pred  # for evaluating prediction
#
#     # TODO: This rescale factor only works for the 3D Lorenz-63 System and has
#     # TODO: to be changed for all other Systems! just plot the log-log plot and
#     # TODO: then change the rest of the code accordingly
#     lorenz_rescale_factor = 8.5
#
#     # adapt parameters to input size:
#     r_min *= traj.std(axis=0).mean() / lorenz_rescale_factor
#     r_max *= traj.std(axis=0).mean() / lorenz_rescale_factor
#     r_steps *= traj.std(axis=0).mean() / lorenz_rescale_factor
#
#     nr_points = float(traj.shape[0])
#     radii = np.arange(r_min, r_max, r_steps)
#
#     tree = scipy.spatial.cKDTree(traj)
#     N_r = np.array(tree.count_neighbors(tree, radii), dtype=float) / nr_points
#     N_r = np.vstack((radii, N_r))
#
#     # linear fit based on loglog scale, to get slope/dimension:
#     slope, intercept = np.polyfit(np.log(N_r[0]), np.log(N_r[1]), deg=1)[0:2]
#     dimension = slope
#
#     ###plotting
#     if plot:
#         plt.loglog(N_r[0], N_r[1], 'x', basex=10., basey=10.)
#         plt.title('loglog plot of the N_r(radius), slope/dim = ' + str(slope))
#         plt.show()
#     return dimension


# def lyapunov(reservoir, threshold=int(10),
#              plot=False, print_switch=False, test_measure=False):
#     """
#     Calculates the maximal Lyapunov Exponent of reservoir.y_pred (or reservoir.y_test),
#     by estimating the time derivative of the mean logarithmic distances of
#     former next neighbours. Stores it in reservoir.lyapunov (reservoir.lyapunov_test)
#     Only values for tau_min/max are used for calculating the slope!
#
#     Since the attractor has a size of roughly 20 [log(20) ~ 3.] this
#     distance reaches a maximum after a certain time, approximately
#     after 4. time units [time_units = dt*steps]
#     Therefore the default values are choosen to be dt dependent as in
#     ###Definition of taus:
#
#     tau_min/max are given in units of steps
#     plot to check for correct average
#     """
#     """
#     REMINDER:
#     remove the loop over taus, since the slope is calculated with single
#     values only
#     """
#     ###Definition of taus:
#     tau_min = int(0.5 / reservoir.dt)
#     tau_max = int(3.8 / reservoir.dt)
#     taus = np.arange(tau_min, tau_max,
#                      10)  # taus = np.array([tau_min, tau_max])
#
#     if test_measure:
#         traj = reservoir.y_test  # for measure assessing
#     else:
#         traj = reservoir.y_pred  # for evaluating prediction
#
#     tree = scipy.spatial.cKDTree(traj)
#     nn_index = tree.query(traj, k=2)[1]
#
#     # drop all elements in nn_index lists where the neighbour is:
#     # 1. less than threshold time_steps away
#     # 2. where we cannot calculate the neighbours future in tau_max time_steps:
#
#     # contains indices of points and the indices of their nn:
#     reservoir.nn_index = nn_index[
#         np.abs(nn_index[:, 0] - nn_index[:, 1]) > threshold]
#
#     nn_index = nn_index[nn_index[:, 1] + tau_max < traj.shape[0]]
#     nn_index = nn_index[nn_index[:, 0] + tau_max < traj.shape[0]]
#
#     # Calculate the largest Lyapunov exponent:
#     # for storing the results:
#     Sum = []
#     # loop over differnt tau, to get a functional dependence:
#     for tau in taus:
#         # print(tau)
#
#         S = []  # the summed values for all basis points
#
#         # loop over every point in the trajectory, where we can calclutate
#         # the future in tau_max time_steps:
#         for point, nn in nn_index:
#             S.append(np.log(np.linalg.norm(traj[point + tau] - traj[
#                 nn + tau])))  # add one points average s to S
#
#             # since there is no else, we only avg over the points that have
#             # points in their epsilon environment
#         Sum.append((tau * reservoir.dt, np.array(S).mean()))
#     Sum = np.array(Sum)
#
#     slope = (Sum[-1, 1] - Sum[0, 1]) / (Sum[-1, 0] - Sum[0, 0])
#     if plot:
#         plt.title('slope: ' + str(slope))
#         plt.plot(Sum[:, 0], Sum[:, 1])
#         plt.plot(Sum[:, 0], Sum[:, 0] * slope)
#         plt.xlabel('time dt*tau[steps]')
#         plt.ylabel('log_dist_former_neighbours')
#         # plt.plot(Sum[:,0], Sum[:,0]*slope + Sum[0,0])
#         plt.show()
#
#     return slope


# def W_out_distr(self):
#     """
#     Shows a histogram of the fitted parameters of self.w_out, each output
#     dimension in an other color
#     """
#     f = plt.figure(figsize=(10, 10))
#     for i in np.arange(self.y_dim):
#         plt.hist(self.w_out[i], bins=30, alpha=0.5, label='w_out[' + str(i) + ']')
#     plt.legend(fontsize=10)
#     f.show()

# def calc_strength(self):
#     """
#     Calculate the absolute in and out strength of nodes in self.network
#     and its respective average.
#     Stores them in :self.in_strength, self.avg_in_strength, self.out_strength, 
#     self.avg_out_strength
#     """
#     self.in_strength = np.abs(self.network).sum(axis=0)
#     self.avg_in_strength = self.in_strength.mean()
#     self.out_strength = np.abs(self.network).sum(axis=1)
#     self.avg_out_strength = self.out_strength.mean()

# def clustering_coeff(reservoir):
#     """
#     clustering coefficient for each node and returns it.
#     """
#     reservoir.calc_binary_network()
#     network = reservoir.binary_network
#     k = network.sum(axis=0)
#     C = np.diag(network @ network @ network) / k * (k - 1)
#     reservoir.clustering_coeff = C


# def calc_tt(reservoir, flag='bool', split=0.1):
#     """
#     selects depending on if the abs(entry) of reservoir.w_out is one of the
#     largest, depending on split.
#     If split is negative the abs(entry) smallest are selected depending
#     on flag:
#     - 'bool': reservoir.w_out.shape with True/False
#     - 'bool_1d': is a projection to 1d
#     - 'arg': returns args of the selection
#
#     """
#     if reservoir.r_squared:
#         print('no tt_calc for r_squared implemented yet')
#     else:
#         absolute = int(reservoir.ndim * split)
#         n = reservoir.ydim * reservoir.ndim  # dof in w_out
#         top_ten_bool = np.zeros(n, dtype=bool)  # False array
#         arg = np.argsort(
#             np.reshape(np.abs(reservoir.W_out), -1))  # order of abs(w_out)
#         if absolute > 0:
#             top_ten_bool[arg[-absolute:]] = True  # set largest entries True
#             top_ten_arg = np.argsort(np.max(np.abs(reservoir.W_out), axis=0))[
#                           -absolute:]
#         elif absolute < 0:
#             top_ten_bool[arg[:-absolute]] = True  # set largest entries True
#             top_ten_arg = np.argsort(np.max(np.abs(reservoir.W_out), axis=0))[
#                           :-absolute]
#         else:
#             top_ten_arg = np.empty(0)
#
#         top_ten_bool = np.reshape(top_ten_bool,
#                                   reservoir.W_out.shape)  # reshape to original shape
#         top_ten_bool_1d = np.array(top_ten_bool.sum(axis=0),
#                                    dtype=bool)  # project to 1d
#
#         if flag == 'bool':
#             return top_ten_bool
#         elif flag == 'bool_1d':
#             return top_ten_bool_1d
#         elif flag == 'arg':
#             return top_ten_arg


# def weighted_clustering_coeff_onnela(reservoir):
#     """
#     Calculates the weighted clustering coefficient of abs(self.network)
#     according to Onnela paper from 2005.
#     Replacing NaN (originating from division by zero (degree = 0,1)) with 0.
#     Returns weighted_cc.
#     """
#     k = reservoir.binary_network.sum(axis=0)
#     # print(k)
#     network = abs(reservoir.network) / abs(reservoir.network).max()
#
#     network_cbrt = np.cbrt(network)
#     weighted_cc = np.diag(network_cbrt @ network_cbrt @ network_cbrt) / \
#                   (k * (k - 1))
#     # assign 0. to infinit values:
#     weighted_cc[np.isnan(weighted_cc)] = 0.
#     return weighted_cc


#    def calc_covar_rank(reservoir, flag='train'):
#        """
#        Calculated the covarianc rank of the squared network dynamics matrix self.r
#        (or self.r_pred) and stores it in self.covar_rank
#        """
#        """
#        Does not calculate the actual covariance matrix!! Fix befor using
#        """
#        if flag == 'train':
#            res_dyn = self.r
#        elif flag == 'pred':
#            res_dyn = self.r_pred
#        else:
#            raise Exception("wrong covariance flag")
#        covar = np.matmul(res_dyn.T, res_dyn)
#        #self.covar_rank = np.linalg.matrix_rank(covar)
#        print(np.linalg.matrix_rank(covar))


pass  # TODO: Add below to ESNWrapper
# def remove_nodes(reservoir, split):
#     """
#     This method removes nodes from the network and w_in according to split,
#     updates avg_degree, spectral_radius,
#     This new reservoir is returned
#     split should be given as a list of two values or a float e [-1. and 1.]
#     example: split = [-.3, 0.3]
#     """
#     if type(split) == list:
#         if len(split) < 3:
#             pass
#         else:
#             raise Exception('too many entries in split. length: ', len(split))
#     elif type(split) == float and split >= -1. and split <= 1.:
#         split = [split]
#     else:
#         raise Exception('values in split not between -1. and 1., type: ',
#                         type(split))
#
#     remaining_size = sum(np.abs(split))
#
#     new = ESN(sys_flag=reservoir.sys_flag,
#               network_dimension=int(
#                           round(reservoir.ndim * (1 - remaining_size))),
#               input_dimension=3, output_dimension=3,
#               type_of_network=reservoir.type, dt=reservoir.dt,
#               training_steps=reservoir.training_steps,
#               prediction_steps=reservoir.prediction_steps,
#               discard_steps=reservoir.discard_steps,
#               regularization_parameter=reservoir.reg_param,
#               spectral_radius=reservoir.spectral_radius,
#               avg_degree=reservoir.avg_degree,
#               epsilon=reservoir.epsilon,
#               # activation_function_flag=reservoir.activation_function_flag,
#               w_in_sparse=reservoir.W_in_sparse,
#               w_in_scale=reservoir.W_in_scale,
#               bias_scale=reservoir.bias_scale,
#               normalize_data=reservoir.normalize_data,
#               r_squared=reservoir.r_squared)
#     # gather to be removed nodes arguments in rm_args:
#     rm_args = np.empty(0)
#     for s in split:
#         rm_args = np.append(calc_tt(reservoir, flag='arg', split=s), rm_args)
#         # print(s, rm_args.shape)
#
#     # rows and columns of network are deleted according to rm_args:
#     new.network = np.delete(np.delete(reservoir.network, rm_args, 0), rm_args,
#                             1)
#     # the new average degree is calculated:
#     new.calc_binary_network()
#     new.avg_degree = new.binary_network.sum(axis=0).mean(axis=0)
#     # the new spectral radius is calculated:
#     new.network = scipy.sparse.csr_matrix(new.network)
#     try:
#         eigenvals = scipy.sparse.linalg.eigs(new.network,
#                                              k=1,
#                                              v0=np.ones(new.n_dim),
#                                              maxiter=1e3*new.n_dim)[0]
#         new.spectral_radius = np.absolute(eigenvals).max()
#
#         # try:
#         #     eigenvals = scipy.sparse.linalg.eigs(new.network, k=1, which='LM')[0]
#         #     new.spectral_radius = np.absolute(eigenvals).max()
#         # except:
#         #     print('eigenvalue calculation failed!, no spectral_radius assigned')
#
#         new.network = new.network.toarray()
#
#     except ArpackNoConvergence:
#         print('Eigenvalue in remove_nodes could not be calculated!')
#         raise
#
#     # Adjust w_in
#     new._w_in = np.delete(reservoir.W_in, rm_args, 0)
#     # pass x,y to new_reservoir
#     new.x_train = reservoir.x_train
#     new.x_discard = reservoir.x_discard
#     new.y_test = reservoir.y_test
#     new.y_train = reservoir.y_train
#
#     return new
