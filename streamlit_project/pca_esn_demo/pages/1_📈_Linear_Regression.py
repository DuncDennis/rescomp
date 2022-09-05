"""A streamlit app to demonstrate PCA is conjunction with Echo State Networks - Linear Reg."""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import streamlit as st

st.set_page_config("Linear Regression", page_icon="📈")

st.header("Linear Regression")

st.markdown(
    r"""
    In linear regression one tries to find a linear relationship between some explanatory 
    variables $\boldsymbol{r}$, and some dependent variables $\boldsymbol{y}$. 
    """
)

st.latex(r"""
\begin{aligned}
    \boldsymbol{r}&: \text{Explanatory variables of dimension } r_\text{dim} \\
    \boldsymbol{y}&: \text{Dependent variables of dimension } y_\text{dim}
\end{aligned}
""")

st.markdown(
    r"""
    The aim is to find the parameters of $y_\text{dim} \times r_\text{dim}$ matrix $W_\text{out}$, 
    and the $y_\text{dim}$ offset-vector $\boldsymbol{w}$, so that the relationship: 
    """
)

st.latex(r""" 
\begin{aligned}
    \boldsymbol{y}& = W_\text{out}\boldsymbol{r} + \boldsymbol{w} = W_\text{out}^*[\boldsymbol{r}, 
    1] = W_\text{out}^* \boldsymbol{r}^*
\end{aligned}
""")

st.markdown(
    r"""
    holds "as good as possible". We can incorporate the offset-vector $\boldsymbol{w}$ into the 
    matrix, by searching a  $r_\text{y} \times (r_\text{dim} + 1)$ matrix $W^*_\text{out}$, 
    where we also extend the $\boldsymbol{r}$ by one dimension with the constant value of 1. 
    Mathematically, one tries to find $W^*_\text{out}$, so that the squared difference, summed over 
    all data points $i$ is minimized: 
    """
)

st.latex(r""" 
\begin{aligned}
    \sum_{\text{sample: }i} ||\boldsymbol{y}_i& - W_\text{out}^* \boldsymbol{r}_i^*||^2
\end{aligned}
""")


st.markdown(
    r"""
    $W^*_\text{out}$ can be calculated via: 
    """
)

st.latex(r"""
\begin{aligned}
    W_\text{out}^* = (R^T R)^{-1} R^T Y
\end{aligned}
""")

st.markdown(
    r"""
    Where $R$ and $Y$ are the matrices corresponding to the collection of data points of 
    $\boldsymbol{r}^*_i$ and $\boldsymbol{y}_i$ respectively. 
    """
)

st.markdown(
    r"""
    The above calculation assumes that the inverse of the 
    [moment matrix](https://en.wikipedia.org/wiki/Moment_matrix) is invertible. 
    """
)

st.latex(r"""
\begin{aligned}
    \text{Moment matrix} = R^T R \\
    (R^T R)_{j, k} = \sum_{\text{sample: }i} r^*_{i, j} r^*_{i, k} 
\end{aligned}
""")

st.markdown(
    r"""
    It turns out, $R^T R$ is only nicely invertible, if all features (dimensions of $r^*$) are
    not linear combinations of each other 
    (no [Multicollinearity](https://en.wikipedia.org/wiki/Moment_matrix)), e.g. if: 
    """
)

st.latex(r"""
\begin{aligned}
    (\hat{e}_k \cdot \boldsymbol{r}^*) = a + b (\hat{e}_j \cdot \boldsymbol{r}^*), k \neq j
\end{aligned}
""")

st.markdown(
    r"""
    holds for some dimensions $k, j$ in $\boldsymbol{r}$, there is multicollinearity and the the moment
    matrix can not be inverted. 
    """
)

st.subheader("Issue of multicollinearity")

st.markdown(
    r"""
    **How multicollinearity can be detected:** 
    - **Condition of data matrix** $X$: [Condition number](https://en.wikipedia.org/wiki/Condition_number)
    - **Perturbing the data with noise and see how much the coefficients change**:
    """
)

st.markdown(
    r"""
    What to do against multicollinearity: 
    - Mean center the predictor variables (?)
    - Standardize your independent variables
    - Shapley value? [Shapley value](https://en.wikipedia.org/wiki/Shapley_value)
    - Tikhonov regularization / ridge regression [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression)
    - Principal component regression [Principal_component_regression](https://en.wikipedia.org/wiki/Principal_component_regression) 
    """
)

st.markdown(
    r"""
    **Principal component regression (PCR):**
    - Instead of using the explanatory variables directly, the principal components are used as
        the regressors. 
    - If one uses only a subset of PCs, PCR is a kind of regularized linear regression. 
    - Major use case is: overcoming multicollinearity. 
    - Steps: 
        - Perform pca on the data matrix. (maybe only select a subset of components). 
        - Perform linear regression (or ridge regression?). 
        - Transform w_out back to get the w_out for the original features. 
    - Steps in detail: 
        - Datapreprocessing: Assume Y and X have been centered. 
        - Perform PCA on centered $X$. 
            - $X = U \Delta V^T$ is the singular value decomposition. With \Delta are the singular
                values (only on diagonal). $U$ and $V$ are both orthonormal sets of vectors, 
                denoting the left and right singular vectors of $X$. 
            - $V \Lambda V^T = V \Delta^2 V^T$ gives the spectral decomposition of $X^T X$. $\Lambda$ 
                contains the principal values on the diagonal. 
        - PCA Estimator: $W_\text{out, pca} = (R_\text{pca}^TR_\text{pca})^{-1} R_\text{pca}^TY$ 
            is the regressor for the pca transformed variables. $W_\text{out}$ for the original 
            variables can be estimated via $W_\text{out} = VW_\text{out, pca}$
    - Notes: 
        - If all components are considered, the PCA estimator is the same as the OLS estimate. 
    - Applications: 
        - Adressing multicollinearity: Exclude the columns that are multicollinear. 
        - Regularization effect: 
    """
)

st.subheader("Demonstration: ")


def get_r_states(r_dim: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """Get sample r_states.

    Args:
        r_dim: Nr of explanatory variables.
        n_samples: Nr of samples.
        seed: Seed for random number generator.

    Returns:
        The r_states of shape (n_samples, r_dim).
    """
    rng = np.random.default_rng(seed)
    r_states = rng.random((n_samples, r_dim))
    return r_states


def get_out_states(r_states: np.ndarray) -> np.ndarray:
    """Calculate the output states as a simple linear function of the input states.

    Args:
        r_states: The input states of shape (n_samples, r_dim).

    Returns:
        The output states of shape (n_samples, 1).
    """

    offset = 2
    out_states = np.sum(r_states, axis=-1)[:, np.newaxis] + offset
    return out_states


def get_multicol_r_states(r_states: np.ndarray,
                          nr_of_colinear_cols: int,
                          error_scale: float = 0.0,
                          seed: int = 0,
                          ) -> np.ndarray:
    """Take original_r_states and create a number of multi-colinear columns.

    Create the multi-colinearity by overwriting the ith column with the
    (r_dim - nr_of_colinear_cols + i)th column plus a normal error with scale "error_scale".

    Args:
        r_states: The original r_states of shape (n_samples, r_dim).
        nr_of_colinear_cols: The number of multi-colinear columns you wish to create. Has to be
                             smaller than r_dim - 1.
        error_scale: The scale of the gaussian error you add to the multi-colinear scale.
        seed: Seed for random number generator.

    Returns:
        The r_states with multi-colinearity introduced of shape (n_samples, r_dim).
    """

    if nr_of_colinear_cols >= r_states.shape[1]:
        raise ValueError("Nr of colinear columns has to be smaller than r_dim - 1. ")

    rng = np.random.default_rng(seed)
    error_values = rng.standard_normal(nr_of_colinear_cols) * error_scale
    r_states_m_col = r_states.copy()
    if nr_of_colinear_cols > 0:
        for i in range(nr_of_colinear_cols):
            r_states_m_col[:, i] = r_states_m_col[:, (r_dim - nr_of_colinear_cols) + i] + \
                                   error_values[i]
    return r_states_m_col


def get_r_gen_states(r_states: np.ndarray) -> np.ndarray:
    """Append a one to all r_states. """
    n_samples = r_states.shape[0]
    return np.hstack((r_states, np.ones(n_samples)[:, np.newaxis]))


def get_moment_matrix(r_gen_states: np.ndarray) -> np.ndarray:
    """Calculate the moment matrix R^T R."""
    return r_gen_states.T @ r_gen_states


def linear_regression(r_gen_states: np.ndarray, out_states: np.ndarray) -> np.ndarray:
    """Perform ordinary linear regression.

    Args:
        r_gen_states: Matrix of explanatory variables of shape (n_samples, r_gen_dim).
        out_states: Matrix of dependent variables of shape (n_samples, out_dim).

    Returns:
        The W_out matrix that fits out_states = W_out @ r_gen_states of shape (y_dim, r_gen_dim).
    """
    return np.linalg.solve(
        r_gen_states.T @ r_gen_states,
        r_gen_states.T @ out_states).T


def ridge_regression(r_gen_states: np.ndarray, out_states: np.ndarray,
                     reg_param: float = 1e-7) -> np.ndarray:
    """Perform ordinary linear regression.

    Args:
        r_gen_states: Matrix of explanatory variables of shape (n_samples, r_gen_dim).
        out_states: Matrix of dependent variables of shape (n_samples, out_dim).
        reg_param: Small regularization parameter for ill-conditioned / multi-collinear
                   r_gen_states.

    Returns:
        The W_out matrix that fits out_states = W_out @ r_gen_states of shape (y_dim, r_gen_dim).
    """
    return np.linalg.solve(
        r_gen_states.T @ r_gen_states + reg_param + np.eye(r_gen_states.shape[1]),
        r_gen_states.T @ out_states).T


# Get R states:
cols = st.columns(2)
with cols[0]:
    n_samples = st.number_input("n_samples", value=10000)
with cols[1]:
    r_dim = st.number_input("r_dim", value=10)

seed = 0
r_states = get_r_states(r_dim, n_samples, seed=seed)
r_gen_states = get_r_gen_states(r_states)
out_states = get_out_states(r_states)
moment_matrix = get_moment_matrix(r_gen_states)

cond = np.linalg.cond(r_states)
cond

rank = np.linalg.matrix_rank(r_states)
rank

# Get multi-colinearity:
cols = st.columns(2)
with cols[0]:
    nr_of_colinear_cols = st.number_input("Nr of multi-colinear columns",
                                          value=1,
                                          min_value=0,
                                          max_value=r_dim-1)
with cols[1]:
    error_scale = st.number_input("Error scale",
                                  value=0.0,
                                  step=0.01,
                                  min_value=0.0)

r_states_m_col = get_multicol_r_states(r_states, nr_of_colinear_cols,
                                       error_scale, seed=seed)

out_states_m_col = get_out_states(r_states_m_col)

cond_m_col = np.linalg.cond(r_states_m_col)
cond_m_col

rank_m_col = np.linalg.matrix_rank(r_states_m_col)
rank_m_col

r_gen_states_m_col = get_r_gen_states(r_states_m_col)
moment_matrix_m_col = get_moment_matrix(r_gen_states_m_col)


# Fit the data:
# 1. Ordinarly linear Regression:
w_out_bias = linear_regression(r_gen_states, out_states)
w_out_bias

w_out_bias_m_col = linear_regression(r_gen_states_m_col, out_states_m_col)
w_out_bias_m_col
