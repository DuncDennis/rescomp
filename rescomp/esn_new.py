"""Echo State Networks"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rescomp import utilities


class ESNCore(ABC):
    """The abstract base class for Echo State Networks."""

    r_dim: int  # reservoir state dimension
    r_gen_dim: int  # generalized reservoir state dimension
    x_dim: int  # input dimension
    y_dim: int  # output dimension

    w_out: np.ndarray  # Output matrix of shape (y_dim, r_gen_dim).

    leak_factor: float  # Leak factor in the update equation.
    node_bias: np.ndarray  # Node bias for each reservoir node.

    @abstractmethod
    def activation_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method for the element wise activation function"""

    @abstractmethod
    def output_to_input_fct(self, x: np.ndarray) -> np.ndarray:
        """Abstract method to connect the output back to the next input."""

    @abstractmethod
    def input_to_reservoir_fct(self, x: np.ndarray) -> np.ndarray:
        """Abstract method to connect the input with the reservoir."""

    @abstractmethod
    def internal_reservoir_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method to internally update the reservoir state."""

    @abstractmethod
    def r_to_r_gen_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method for the reservoir to generalized reservoir state function. """

    def set_dimensions(self, x_dim: int, y_dim: int, r_dim: int) -> None:
        """Set the input, reservoir and output_dimension. """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim

    def reservoir_update(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """The reservoir update equation.

        Args:
            x: The input of shape (x_dim, ).
            r: The previous reservoir state of shape (r_dim, ).

        Returns:
            The next reservoir state of shape (r_dim, ).
        """

        input_to_res = self.input_to_reservoir_fct(x)
        res_to_res = self.internal_reservoir_fct(r)
        activation_fct_argument = input_to_res + res_to_res + self.node_bias
        next_r = self.leak_factor * r + (1 - self.leak_factor) * activation_fct_argument(
            activation_fct_argument)
        return next_r

    def r_gen_to_output(self, r_gen: np.ndarray) -> np.ndarray:
        """Function to calculate the output from the generalized reservoir state using W_out. """
        output = self.w_out @ r_gen
        return output


# class ActivationFunctionMixin(ABC):
#     """An abstract base class for all activation function mixins. """
#
#     @abstractmethod
#     def activation_fct(self, r: np.ndarray) -> np.ndarray:
#         """Abstract method for the element wise activation function"""


class OutputToInputMixin(ABC):
    """An abstract base class for all outout to input mixins. """

    @abstractmethod
    def output_to_input_fct(self, x: np.ndarray) -> np.ndarray:
        """Abstract method to connect the output back to the next input."""


class InputToReservoirMixin(ABC):
    """An abstract base class for all input to reservoir mixins. """

    @abstractmethod
    def input_to_reservoir_fct(self, x: np.ndarray) -> np.ndarray:
        """Abstract method to connect the input with the reservoir."""


class InternalReservoirMixin(ABC):
    """An abstract base class for all internal reservoir mixins. """

    @abstractmethod
    def internal_reservoir_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method to internally update the reservoir state."""


class RRGenMixin(ABC):
    """An abstract base class for all r to r_gen mixins. """

    @abstractmethod
    def r_to_r_gen_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method for the reservoir to generalized reservoir state function. """


class NodeBiasMixin:
    """Mixin class to set the node_bias. """

    _node_bias_flag_synonyms = utilities._SynonymDict()
    _node_bias_flag_synonyms.add_synonyms(0, ["no_bias"])
    _node_bias_flag_synonyms.add_synonyms(1, ["random_bias"])
    _node_bias_flag_synonyms.add_synonyms(2, ["constant_bias"])

    def set_node_bias(self, r_dim: int, node_bias_opt: str = "no_bias", bias_scale: float = 1.0
                      ) -> None:
        """Set the node bias (self.node_bias) of the ESN.

        Args:
            r_dim: The reservoir dimension.
            node_bias_opt: The node bias option being either "no_bias", "random_bias" or
                           "constant_bias".
            bias_scale:

        """
        node_bias_flag = self._node_bias_flag_synonyms.get_flag(node_bias_opt)
        if node_bias_flag == 0:
            self.node_bias = np.zeros(0)
        elif node_bias_flag == 1:
            self.node_bias = bias_scale * np.random.uniform(low=-1.0, high=1.0, size=r_dim)
        elif node_bias_flag == 2:
            self.node_bias = bias_scale * np.ones(r_dim)


class LeakFactorMixin:
    """Mixin class to set the leak factor (self.leak_factor) of the ESN. """
    def set_leak_factor(self, leak_factor: float = 0.0) -> None:
        """Set the leak factor of the ESN.

        Args:
            leak_factor: The leak factor between 0 and 1.

        """
        if leak_factor < 0 or leak_factor > 1:
            raise ValueError("Leak factor must be between 0 and 1.")
        self.leak_factor = leak_factor


class ActivationFunctionMixin:
    """Mixin to define the reservoir node activation functions. """
    _act_fct_flag_synonyms = utilities._SynonymDict()
    _act_fct_flag_synonyms.add_synonyms(0, ["tanh"])
    _act_fct_flag_synonyms.add_synonyms(1, ["sigmoid"])
    _act_fct_flag_synonyms.add_synonyms(2, ["relu"])
    _act_fct_flag_synonyms.add_synonyms(3, ["linear"])

    def set_activation_function(self, act_fct_opt: str = "tanh") -> None:
        """Set the reservoir node activation function.

        Args:
            act_fct_opt: The activation function option: "tanh", "sigmoid", "relu", "linear".

        """
        act_fct_flag = self._act_fct_flag_synonyms.get_flag(act_fct_opt)
        if act_fct_flag == 0:
            self._act_fct = utilities.tanh
        elif act_fct_flag == 1:
            self._act_fct = utilities.sigmoid
        elif act_fct_flag == 2:
            self._act_fct = utilities.relu
        elif act_fct_flag == 3:
            self._act_fct = utilities.linear
        else:
            raise ValueError

    def activation_fct(self, x: np.ndarray) -> np.ndarray:
        return self._act_fct(x)


class RRgenMixin:
    _r_to_r_gen_synonyms = utilities._SynonymDict()
    _r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
    _r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
    _r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
    _r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
    _r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
    _r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
    _r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt: str = "linear"):
        pass

class SimpleOutputToInputMixin(OutputToInputMixin):
    """The standard output to input option where the last output becomes the next input."""
    def output_to_input_fct(self, y: np.ndarray) -> np.ndarray:
        """Just the identity function.

        Args:
            y: The previous reservoir prediction output of shape (y_dim, ).

        Returns:
            x: The next reservoir input of shape (y_dim = x_dim, )
        """
        return y


class SimpleInputToReservoirMixin(InputToReservoirMixin):
    """The standard input to reservoir coupling with an input matrix w_in."""
    def input_to_reservoir_fct(self, x: np.ndarray) -> np.ndarray:
        return x


class SimpleInternalReservoirMixin(InternalReservoirMixin):
    def internal_reservoir_fct(self, r: np.ndarray) -> np.ndarray:
        return r


class SimpleRRGENMixin(RRGenMixin):
    def r_to_r_gen_fct(self, r: np.ndarray) -> np.ndarray:
        return r


def get_func(x):
    def a(y): return x*y
    return a

class SimpleESN(ActivationFunctionMixin,
                SimpleOutputToInputMixin,
                SimpleInputToReservoirMixin,
                SimpleInternalReservoirMixin,
                SimpleRRGENMixin,
                ESNCore):

    def __init__(self) -> None:
        super().__init__()

    @property
    def leak_factor(self):
        return self._leak_factor

    @leak_factor.setter
    def leak_factor(self, leak_factor: float):
        self._leak_factor = leak_factor

    def build(self, x_dim, r_dim, model) -> None:
        """tada"""
        ESNCore.set_dimensions(self, x_dim=x_dim, r_dim=r_dim, y_dim=x_dim)

        self._leak_factor = 0.1

        ActivationFunctionMixin.set_activation_function(self, "tanh")
        self.model = model

        self.model2 = get_func(2)

def model(x): return x

import streamlit as st

@st.experimental_memo
def st_test():
    esn = SimpleESN()
    esn.build(5, 4, model=model)
    esn.leak_factor = 5
    return esn


if __name__ == "__main__":
    esn = st_test()
    st.write(esn.build)
    st.write(esn.__dict__)
    if st.button("run"):
        esn.output_to_input_fct()
