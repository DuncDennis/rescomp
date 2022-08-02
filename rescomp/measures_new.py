"""Measures and other analysis functions useful for RC."""

from __future__ import annotations

from typing import Callable

import numpy as np


def largest_lyapunov_exponent(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    steps: int = int(1e3),
    part_time_steps: int = 15,
    steps_skip: int = 50,
    dt: float = 1.0,
    initial_pert_direction: np.ndarray | None = None,
    return_convergence: bool = False,
) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent given an iterator function.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        steps: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        steps_skip: Number of renormalization steps to perform, before tracking the log divergence.
                Avoid transients by using steps_skip.
        dt: Size of time step.
        initial_pert_direction:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy array of
                            the shape (N, ).

    Returns:
        The largest Lyapunov Exponent. If return_convergence is True: The convergence (np.ndarray),
        else just the float value, which is the last value in the convergence.
    """

    x_dim = starting_point.size

    if initial_pert_direction is None:
        initial_pert_direction = np.ones(x_dim)

    initial_perturbation = initial_pert_direction * (deviation_scale / np.linalg.norm(initial_pert_direction))

    log_divergence = np.zeros(steps)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(steps + steps_skip):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        x_pert = x + dx * (deviation_scale / norm_dx)
        if i_n >= steps_skip:
            log_divergence[i_n - steps_skip] = np.log(norm_dx / deviation_scale)

    if return_convergence:
        return np.array(np.cumsum(log_divergence) / (np.arange(1, steps + 1) * dt * part_time_steps))
    else:
        return float(np.average(log_divergence) / (dt * part_time_steps))
