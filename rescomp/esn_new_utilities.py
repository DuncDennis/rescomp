import numpy as np


# Activation functions:
def sigmoid(x: np.ndarray) -> np.ndarray:
    """The sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """The RELU activation function."""
    return x * (x > 0)


def linear(x: np.ndarray) -> np.ndarray:
    """No activation function."""
    return x


def tanh(x: np.ndarray) -> np.ndarray:
    """The tanh activation function."""
    return np.tanh(x)


# R to R gen functions:
def linear_rrgen(r: np.ndarray) -> np.ndarray:
    """No r to r gen function.
    
    Args:
        r: The input of shape (r_dim, ). 

    Returns:
        The generalized reservoir state of shape (r_dim, ).
    """
    return r


def linear_square_rrgen(r: np.ndarray) -> np.ndarray:
    """Square the r vector and stack on top of the original. 
    
    Args:
        r: The input of shape (r_dim, ). 

    Returns:
        The generalized reservoir state of shape (2 * r_dim, ).
    """
    return np.hstack((r, r ** 2))


def output_bias_rrgen(r: np.ndarray) -> np.ndarray:
    """Add a one to the r vector to enable the learning of an offset.
    
    Args:
        r: The input of shape (r_dim, ). 

    Returns:
        The generalized reservoir state of shape (r_dim + 1, ).
    """
    return np.hstack((r, 1))


def linear_square_bias_rrgen(r: np.ndarray) -> np.ndarray:
    """Stack the square and 1 on top of r.
    
    Args:
        r: The input of shape (r_dim, ). 

    Returns:
        The generalized reservoir state of shape (2 * r_dim + 1, ).
    """
    return np.hstack((np.hstack((r, r ** 2)), 1))


def linear_square_alternate_rrgen(r: np.ndarray) -> np.ndarray:
    """Square every second entry in the r vector.

    Args:
        r: The input of shape (r_dim, ).

    Returns:
        The generalized reservoir state of shape (r_dim, ).
    """
    r_gen = np.copy(r).T
    r_gen[::2] = r.T[::2] ** 2
    return r_gen.T


def identity_output_to_input(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The identity output to input function.

    The second argument (the previous input) is required but not used.

    Args:
        y: The output of the reservoir.
        x: The previous input to the reservoir (not required).

    Returns:
        The next input to the reservoir.
    """
    return y


def difference_output_to_input(y: np.ndarray, x: np.ndarray, dt: float = 1.0
                               ) -> np.ndarray:
    """The difference output to input function option.

    Use if the reservoir is trained on the difference.

    Args:
        y: The output of the reservoir.
        x: The previous input to the reservoir.
        dt: The timestep dt.

    Returns:
        The next input to the reservoir.
    """
    return y * dt + x


ACT_FCT_DICT = {
    "tanh": tanh,
    "sigmoid": sigmoid,
    "linear": linear,
    "relu": relu
}


RRGEN_FCT_DICT = {
    "linear": linear_rrgen,
    "output_bias": output_bias_rrgen,
    "linear_and_square": linear_square_rrgen,
    "linear_square_and_bias": linear_square_bias_rrgen,
    "linear_square_alternate": linear_square_alternate_rrgen
}


OUT_TO_INP_FCT_DICT = {
    "identity": identity_output_to_input,
    "difference": difference_output_to_input
}
