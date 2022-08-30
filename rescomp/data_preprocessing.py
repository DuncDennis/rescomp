"""Functions to preprocess/manipulate the data."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def scale_and_shift(time_series: np.ndarray, scale: float | np.ndarray | None = None,
                    shift: float | np.ndarray | None = None
                    ) -> np.ndarray:
    """ Scale and shift a time series.

    First center and normalize the time_series to a std of unity for each axis. Then optionally
    rescale and/or shift the time series.

    Args:
        time_series: The time series of shape (time_steps, sys_dim).
        scale: If None the data is scaled so that the std is 1 for every axis. If float, scale
               every axis so that the std is the scale value. If scale is an array, scale so
               that the std of each axis corresponds to the value in the array.
        shift: If None the data is shifted so that the mean is 0 for each axis. If float, shift
               every axis so that the mean is the shift value. If shift is an array, shift so
               that the mean of each axis corresponds to the value in the array.

    Returns:
        The scaled and shifted time_series.
    """

    sys_dim = time_series.shape[1]

    mean = np.mean(time_series, axis=0)
    std = np.std(time_series, axis=0)

    normalized_and_centered = (time_series - mean) / std

    if scale is not None:
        if type(scale) is float:
            scale_vec = np.ones(sys_dim) * scale
        else:
            scale_vec = scale
    else:
        scale_vec = np.zeros(sys_dim)

    scaled_and_centered = normalized_and_centered * scale_vec

    if shift is not None:
        if type(shift) is float:
            shift_vec = np.ones(sys_dim) * shift
        else:
            shift_vec = shift
    else:
        shift_vec = np.zeros(sys_dim)

    return scaled_and_centered + shift_vec


def add_noise(time_series: np.ndarray,
              noise_scale: float = 0.1,
              seed: int | None = None
              ) -> np.ndarray:
    """Add gaussian noise to a time_series.
    TODO: different noise kinds.
    Args:
        time_series: The input time series of shape (time_steps, sys_dim).
        noise_scale: The scale of the gaussian white noise.
        seed: The seed used to calculate the noise.

    Returns:
        The time series with added noise.
    """

    shape = time_series.shape
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape, scale=noise_scale)
    return time_series + noise


def fourier_transform_surrogate(time_series: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return simple Fourier transform surrogates.

    The seeds for every dimension are constructed by seed_dim = seed + i_dim

    Original source: Haochun Ma. Modified to also work for multidimensional data.

    Args:
        time_series: The time series of shape (timesteps, x_dim).
        seed: The random seed used to randomize the phases.

    Returns:
        A surrogate of the time_series of shape (timesteps, x_dim).
    """

    time_steps, x_dim = time_series.shape

    surrogate_out = np.zeros((time_steps, x_dim))

    for i_x in range(x_dim):
        dim_seed = seed + i_x

        rng = np.random.default_rng(dim_seed)

        y = np.fft.rfft(time_series[:, i_x])
        phi = 2 * np.pi * rng.random(len(y))
        phi[0] = 0.0
        if time_steps % 2 == 0:
            phi[-1] = 0.0
        y = y * np.exp(1j * phi)
        surrogate_out[:, i_x] = np.ascontiguousarray(np.real(np.fft.irfft(y, n=len(time_series))))

    return surrogate_out


def embedding(time_series: np.ndarray,
              embedding_dimension: int,
              delay: int = 1,
              dimension_selection: None | list[int] = None
              ) -> np.ndarray:
    """Embed a timeseries with, select the delay and the dimensions.

    For each input dimension timeseries x(t), the output dimension is
    y(t) = [x(t), x(t - 1 * delay), x(t - 2 * delay), x(t - embedding_dimension * delay)].


    Args:
        time_series: The input time series of shape (timesteps, x_dim).
        embedding_dimension: The number of embedding dimensions to add.
        delay: The time delay to use.
        dimension_selection: A list of ints representing the index of the dimensions to consider.

    Returns:
        The embedded time series of shape (timesteps - delay * embedding_dimension,
        len(dimension_selection)).
    """

    initial_time_steps = time_series.shape[0]

    if dimension_selection is not None:
        time_series = time_series[:, dimension_selection]

    output_time_steps = initial_time_steps - delay * embedding_dimension

    time_series_to_stack = [time_series[:output_time_steps, :], ]
    for i_emb_dim in range(1, embedding_dimension + 1):
        if i_emb_dim * delay + output_time_steps == initial_time_steps:
            time_series_to_stack.append(time_series[i_emb_dim * delay:, :])
        else:
            time_series_to_stack.append(time_series[i_emb_dim * delay: i_emb_dim * delay +
                                                                     output_time_steps, :])

    return np.hstack(time_series_to_stack)


def pca_transform(time_series: np.ndarray) -> np.ndarray:
    """Perform a pca transform the time_series.

    Args:
        time_series: The input time series of shape (timesteps, x_dim).


    Returns:
        The pca transformed time series of shape (timesteps, x_dim).
    """
    return PCA().fit_transform(time_series)
