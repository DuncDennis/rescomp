"""A streamlit app to demonstrate PCA is conjunction with Echo State Networks - Linear Reg."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import streamlit as st
from streamlit_project.app_fragments import streamlit_utilities as utils

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
    \sum_{\text{sample: }i} \|\boldsymbol{y}_i& - W_\text{out}^* \boldsymbol{r}_i^*\|^2
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
    (no [Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)), e.g. if: 
    """
)

st.latex(r"""
\begin{aligned}
    (\hat{e}_k \cdot \boldsymbol{r}^*) = a + b (\hat{e}_j \cdot \boldsymbol{r}^*), k \neq j
\end{aligned}
""")

st.markdown(
    r"""
    holds for some dimensions $k, j$ in $\boldsymbol{r}$, there is multicollinearity and the the 
    moment matrix can not be inverted. The above equation shows a *perfect* multicollinearity. 
    Alternatively multicollinearity can be seen in a loose sense, where the there is only an 
    approximate linear relationship between two or more variables. 
    """
)

st.subheader("Issue of multicollinearity")

st.markdown(
    r"""
    ##### How multicollinearity can be detected:
    - **Condition of moment matrix** $R^TR$ (or of design matrix $R$?) ([Condition number](https://en.wikipedia.org/wiki/Condition_number)):
        - The conditioning of a matrix $A$ is defined as: $\text{cond}(A) = \|A\| \|A^{+}\|$, 
            where $A^+$ is the pseudo-inverse of $A$ and $\|\cdot\|$ can be any matrix norm. 
        - Links: [different matrix norms](https://en.wikipedia.org/wiki/Matrix_norm), 
            [NumPy function](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html)
        - If the data shows multi-collinearity, the conditioning is very large! 
    - **Perturbing the data with noise and see how much the coefficients change**:
        - If the coefficients change a lot, the data is likely to contain multicollinearity. 
        - Note: Perturbing the data with noise already adds a sort of regularization! 
    - **Train on different subsets of the data, and see how the coefficients change**
    - **Rank of moment matrix:** 
        - When perfect multicollinearity exists, the rank of the moment matrix is less than the 
            dimension of the matrix. 
        - Links: [Rank of matrix](https://en.wikipedia.org/wiki/Rank_(linear_algebra)), 
            [NumPy rank function](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html).
    - **Large linear correlations between varaibles**
    """
)

# REGULARIZATIONS TO TACKLE MULTICOLLINEARITY
# ridge regression
st.markdown(
    r"""
    ##### Regularization to tackle multicollinearity: 
    
    **1. Tikhonov regularization / [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression):**
    
    The aim is to find the $W^*_\text{out}$, that minimizes: 
    """
)

st.latex(r""" 
\begin{aligned}
    \gamma\text{Trace}(W_\text{out}^*W_\text{out}^{*T}) + 
    \sum_{\text{sample: }i} \|\boldsymbol{y}_i& - W_\text{out}^* \boldsymbol{r}_i^*\|^2
\end{aligned}
""")

st.markdown(
    r"""
    The additional term tackles the issue of the moment matrix $R^TR$ not being invertable. 
    The minimizing $W^*_\text{out}$ can be calculated with: 
    """
)

st.latex(r""" 
\begin{aligned}
    W_\text{out}^* = (R^T R + \gamma 1)^{-1} R^T Y
\end{aligned}
""")

# Adding noise:
st.markdown(
    r"""
    **2. Adding noise to the training data:**
    
    By adding low amplitude random noise to the training data, the multicollinearity can be 
    overcome, without disturbing the system too much. 
    """
)

# Principal component regression:
st.markdown(
    r"""
    **3. [Principal component regression](https://en.wikipedia.org/wiki/Principal_component_regression):**

    Instead of using the explanatory variables directly, the principal components are used as
    the regressors. 
    
    Steps: 
    - Perform pca on the data matrix. (maybe only select a subset of components). 
    - Perform linear regression on the subset of pca components (or use ridge regression). 
    - Optional: Transform $W^*_\text{out, pca}$ back to get the $W^*_\text{out}$ of the original 
        explanatory variables. 
    """
)

# st.markdown(
#     r"""
#     The
#     - Link: [Ridge regression](https://en.wikipedia.org/wiki/Ridge_regression):
#
#     - Mean center the predictor variables (?)
#     - Standardize your independent variables
#     - Shapley value? [Shapley value](https://en.wikipedia.org/wiki/Shapley_value)
#     - Tikhonov regularization / ridge regression
#     - Principal component regression [Principal_component_regression](https://en.wikipedia.org/wiki/Principal_component_regression)
#     """
# )

# st.markdown(
#     r"""
#     **Principal component regression (PCR):**
#     - Instead of using the explanatory variables directly, the principal components are used as
#         the regressors.
#     - If one uses only a subset of PCs, PCR is a kind of regularized linear regression.
#     - Major use case is: overcoming multicollinearity.
#     - Steps:
#         - Perform pca on the data matrix. (maybe only select a subset of components).
#         - Perform linear regression (or ridge regression?).
#         - Transform w_out back to get the w_out for the original features.
#     - Steps in detail:
#         - Datapreprocessing: Assume Y and X have been centered.
#         - Perform PCA on centered $X$.
#             - $X = U \Delta V^T$ is the singular value decomposition. With \Delta are the singular
#                 values (only on diagonal). $U$ and $V$ are both orthonormal sets of vectors,
#                 denoting the left and right singular vectors of $X$.
#             - $V \Lambda V^T = V \Delta^2 V^T$ gives the spectral decomposition of $X^T X$. $\Lambda$
#                 contains the principal values on the diagonal.
#         - PCA Estimator: $W_\text{out, pca} = (R_\text{pca}^TR_\text{pca})^{-1} R_\text{pca}^TY$
#             is the regressor for the pca transformed variables. $W_\text{out}$ for the original
#             variables can be estimated via $W_\text{out} = VW_\text{out, pca}$
#     - Notes:
#         - If all components are considered, the PCA estimator is the same as the OLS estimate.
#     - Applications:
#         - Adressing multicollinearity: Exclude the columns that are multicollinear.
#         - Regularization effect:
#     """
# )


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
    r_states = rng.random((n_samples, r_dim)) # + 1
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


def noisy_linear_regression(r_gen_states: np.ndarray, out_states: np.ndarray,
                            noise_scale: float = 1e-2,
                            seed: int = 0) -> np.ndarray:
    """Perform ordinary linear regression where noise is added to the input.

    Args:
        r_gen_states: Matrix of explanatory variables of shape (n_samples, r_gen_dim).
        out_states: Matrix of dependent variables of shape (n_samples, out_dim).
        noise_scale: The scale of the white noise added to the training data.
        seed: The random seed used for the white noise.

    Returns:
        The W_out matrix that fits out_states = W_out @ r_gen_states of shape (y_dim, r_gen_dim).
    """
    rng = np.random.default_rng(seed)
    r_gen_states_w_noise = r_gen_states + rng.standard_normal(r_gen_states.shape) * noise_scale
    return np.linalg.solve(
        r_gen_states_w_noise.T @ r_gen_states_w_noise,
        r_gen_states_w_noise.T @ out_states).T


def test_for_multicollinearity(moment_matrix: np.ndarray,
                               title: str | None = None) -> None:
    """Perform some tests to check for multicollinearity and show streamlit elements.

    Args:
        moment_matrix: 2D numpy array being the moment matrix R^TR of the data matrix R.
        title: Optional title.
    """
    rank = np.linalg.matrix_rank(moment_matrix)

    cond = np.linalg.cond(moment_matrix)
    if title is not None:
        st.markdown(f"**{title}**")
    cols = st.columns(2)
    with cols[0]:
        st.write("Condition: ")
        st.write(cond)
    with cols[1]:
        st.write("Rank: ")
        st.write(rank)


def train_on_subset(r_states: np.ndarray, out_states: np.ndarray, seed: int = 1,
                    n_samples_subset: int = 100, n_ens: int = 100) -> np.ndarray:
    """Perform the linear regression on random subsets of r_states and out_states.

    Args:
        r_states: The original r_states of shape (n_samples, r_dim).
        out_states: Matrix of dependent variables of shape (n_samples, out_dim).
        seed: Seed for random number generator.
        n_samples_subset: Nr of samples per subset.
        n_ens: Nr of different subsets.

    Returns:
        W_out for each subset in the shape (n_ens, y_dim, r_gen_dim).
    """

    # Get parameters:
    n_samples, r_dim = r_states.shape
    if n_samples_subset > n_samples:
        raise ValueError("Nr of samples in subset can not be larger than total number of "
                         "samples. ")
    r_gen_dim = r_dim + 1
    out_dim = out_states.shape[1]

    # Configureate rng:
    rng = np.random.default_rng(seed)

    # Calculate w_out for each subset:
    w_out_subsets = np.zeros((n_ens, out_dim, r_gen_dim))
    total_list_of_indices = np.arange(n_samples)
    for i in range(n_ens):
        indices = rng.choice(total_list_of_indices, size=n_samples_subset, replace=False)
        r_states_subset = r_states[indices, :]
        r_gen_states_subset = get_r_gen_states(r_states_subset)
        out_states_subset = out_states[indices, :]
        w_out_subsets[i, :, :] = linear_regression(r_gen_states_subset, out_states_subset)

    return w_out_subsets


def transform_pca_w_out_back(w_out_pca: np.ndarray,
                             r_states: np.ndarray,
                             pca_component_matrix: np.ndarray,
                             ) -> np.ndarray:
    """Transform the w_out matrix after pca transform back to the normal w_out matrix.

    Args:
        w_out_pca: The w_out matrix of the linear regression of r_gen(pca(r_states)) of shape
                   (y_dim, pca_components + 1).
        r_states: The states used to fit the pca of shape (n_samples, r_dim).
        pca_component_matrix: The pca component matrix of shape (pca_components, r_dim).

    Returns:
        The back transformed w_out of shape (y_dim, r_gen_dim).
    """

    r_mean = np.mean(r_states, axis=0)

    w_out_no_bias = w_out_pca[:, :-1] @ pca_component_matrix
    w_out_only_bias = (w_out_pca[:, -1] - w_out_no_bias @ r_mean)[np.newaxis].T

    w_out = np.hstack((w_out_no_bias, w_out_only_bias))
    return w_out

# def do_pcr(r_states: np.ndarray,
#            out_states: np.ndarray,
#            pca_components: int | None = None) -> object:
#     pca = PCA()
#     r_states_pca = pca.fit_transform(r_states)
#     r_gen_states_pca = get_r_gen_states(r_states_pca)
#     moment_matrix_pca = get_moment_matrix(r_gen_states_pca)
#
#     test_for_multicollinearity(moment_matrix_pca)
#
#     w_out_pca = linear_regression(r_gen_states_pca, out_states)
#     st.write(w_out_pca)
#
#     P = pca.components_
#     r_mean = np.mean(r_states, axis=0)
#
#     return w_out_pca, P, r_mean, r_states_pca
#
#     w_out[:, :-1] - w_out_pca[:, :-1] @ P
#     w_out[:, -1] - (w_out_pca[:, -1] - w_out_pca[:, :-1] @ P @ r_mean)


# def add_noise_and_do_reg(r_states: np.ndarray, out_states: np.ndarray,
#                          error_scale: float = 0.1, n_ens: int = 100,
#                          seed: int = 10):
#
#     # Get parameters:
#     n_samples, r_dim = r_states.shape
#     r_gen_dim = r_dim + 1
#     out_dim = out_states.shape[1]
#
#     # Create rng:
#     rng = np.random.default_rng(seed)
#
#     # add error to r_states and transform to r_gen:
#     r_gen_states_w_error = np.zeros((n_ens, n_samples, r_gen_dim))
#     for i in range(n_ens):
#         error = rng.standard_normal((n_samples, r_dim)) * error_scale
#         r_gen_states_w_error[i, :, :] = get_r_gen_states(r_states + error)
#
#     # For each realization do the fit and save w_out:
#     w_out_esn = np.zeros((n_ens, out_dim, r_gen_dim))
#     for i in range(n_ens):
#         r_gen_states = r_gen_states_w_error[i, :, :]
#         # out_states_w_error = out_states + rng.standard_normal(out_dim)
#         out_states_w_error = out_states
#         w_out_esn[i, :, :] = linear_regression(r_gen_states, out_states_w_error)
#
#     return w_out_esn
#     w_out_std = np.std(w_out_esn, axis=0)
#     return w_out_std
#     w_out_avg = np.mean(w_out_esn, axis=0)
#     return w_out_avg


st.header("Demonstration: ")


# NO MULTICOLINEARITY
st.markdown(
    r"""
    ##### No multicollinearity: 
    
    Get explanatory variables $\boldsymbol{r}$ and dependent variables $\boldsymbol{y}$. 
    Specify the number of sampels and the dimension of $\boldsymbol{r}$. 
    $R$ will be random generated by:
    
    `np.hstack((np.random((n_sampels, r_dim)), np.ones(n_samples)[:, np.newaxis]))`
    
    The dependent variables $y$ is one-dimensional and the sum of all expl. variables plus an 
    offset, i.e. 
    
    $\boldsymbol{y} = [\text{offset} + \sum_{\text{dim}: i} r_i, ]^T$. 
    
    The code is: `out_states = np.sum(r_states, axis=-1)[:, np.newaxis] + offset`
    """
)

# Get R states:
cols = st.columns(2)
with cols[0]:
    n_samples = st.number_input("n_samples", value=10000)
with cols[1]:
    r_dim = st.number_input("r_dim", value=10)

seed = 1
r_states = get_r_states(r_dim, n_samples, seed=seed)
r_gen_states = get_r_gen_states(r_states)
out_states = get_out_states(r_states)
moment_matrix = get_moment_matrix(r_gen_states)
# utils.st_line()

st.markdown(
    r"""
    Since all explanatory variables are created randomly, they don't show any correlation or 
    multicollinearity. 
    This can be seen by calculating the **rank** and **condition** of the moment matrix $R^TR$:
    """
)

test_for_multicollinearity(moment_matrix)
# utils.st_line()

# cond = np.linalg.cond(r_states)
# st.write("cond of r_states: ", cond)

st.markdown(
    r"""
    The rank of the moment matrix is $r_{dim} + 1$ (the $+1$ coming from the offset in the linear 
    fit.) Thus we can perform the linear fit without any problems. The output matrix is given as: 
    """
)

w_out = linear_regression(r_gen_states, out_states)
st.markdown(r"$W^*_\text{out}$ (The last entry is the offset $\boldsymbol{w}$):")
st.write(w_out)

st.markdown(
    r"""
    We can further investigate the multicollinearity, by training the system on random subsets 
    of the training data $X, Y$. The $W^*_\text{out}$ will vary between the subsets. The elements
    belonging to multicollinear columns will vary the most. Thus we can calculate the standard
    deviation of the $W^*_\text{out}$ elements over the different subsets. We use $100$ samples 
    per subset and a number of $100$ different subsets.  
    """
)

st.markdown(r"$\text{standard deviation of } W^*_\text{out}$:")

w_out_subset = train_on_subset(r_states, out_states, n_samples_subset=100, n_ens=100)
st.write(np.std(w_out_subset, axis=0, ))

st.markdown(
    r"""
    The standard deviation in $W^*_\text{out}$ is very low between the different subsets. 
    """
)

# Add noise
# st.markdown(
#     r"""
#     We can further investigate the multicollinearity, by adding noise to the design matrix $R$,
#     and observe how much the weights change during training. For an ensemble of $n_\text{ens} = 100$
#     $R$ and $Y$ pairs, with a noise scale (normal noise) of $0.01$, the **standard deviation**
#     within the w_out indices is calculated as:
#     """
# )
# st.markdown(r"$\text{standard deviation of } W^*_\text{out}$:")
# w_out_std = add_noise_and_do_reg(r_states, out_states, error_scale=0.01, n_ens=100)
# st.write(w_out_std)
# st.markdown(
#     r"""
#     The standard deviation in $W^*_\text{out}$ is very low.
#     """
# )

# WITH MULTICOLINEARITY
st.markdown(
    r"""
    ##### With multicollinearity: 
    
    Now we will introduce artificial multicollinearity into the trainings data, by overwriting a
    number variables as a linear function of other variables. 
    
    One can specify the number of variables that are supposed to be written as a linear combination
    of other variables. For imitating non-perfect multicolinearity one can add noise to the linear
    combination. 
    """
)

# Get multi-colinearity:
cols = st.columns(2)
with cols[0]:
    nr_of_colinear_cols = st.number_input("Nr of collinear columns",
                                          value=1,
                                          min_value=0,
                                          max_value=r_dim-1)
with cols[1]:
    error_scale = st.number_input("Error scale",
                                  value=0.01,
                                  step=0.01,
                                  min_value=0.0, format="%f")

r_states_m_col = get_multicol_r_states(r_states, nr_of_colinear_cols,
                                       error_scale, seed=seed)
out_states_m_col = get_out_states(r_states_m_col)
r_gen_states_m_col = get_r_gen_states(r_states_m_col)
moment_matrix_m_col = get_moment_matrix(r_gen_states_m_col)

test_for_multicollinearity(moment_matrix_m_col)

st.markdown(
    r"""
    We can see a huge condition and a rank of: $\text{rank} = r_\text{dim} - 
    \text{nr collinear var}$.
    """
)

w_out_m_col_bias = linear_regression(r_gen_states_m_col, out_states_m_col)
st.markdown(r"$W^*_\text{out}$ (The last entry is the offset $\boldsymbol{w}$):")
st.write(w_out_m_col_bias)

st.markdown(
    r"""
    We can already see that the $W^*_\text{out}$ values for the multicollinear columns are 
    different, than the values for the non-multicollinear columns. 
    
    As before by performing the linear regression on many subsets of $X$ and $Y$, we can 
    observe the standard deviation for each $W^*_\text{out}$ value: 
    """
)

st.markdown(r"$\text{standard deviation of } W^*_\text{out}$:")
w_out_subset_m_col = train_on_subset(r_states_m_col,
                                     out_states_m_col,
                                     n_samples_subset=100,
                                     n_ens=100)

st.write(np.std(w_out_subset_m_col, axis=0))

st.markdown(
    r"""
    We can clearly see, that the standard deviation is very large for entries corresponding to the 
    multicollinear columns. 
    """
)

# ADD REGULARIZATION:
# Ridge regression:
st.markdown(
    r"""
    ##### Adding regularization
    """
)

data_selection = st.selectbox("Data", ["Multicollinear data", "Non-multicollinear data"])
r_states_to_use = r_states.copy() if data_selection == "Non-multicollinear data" else \
    r_states_m_col.copy()
out_states_to_use = out_states.copy() if data_selection == "Non-multicollinear data" else \
    out_states_m_col.copy()
r_gen_states_to_use = get_r_gen_states(r_states_to_use)


st.markdown(
    r"""
    **Ridge Regression**: 
    
    Choose the regularization parameter, and perform the same tests as above:
    """
)

log_reg_param = st.number_input("Log reg. param", value=-7)
reg_param = 10**(log_reg_param)

w_out_ridge = ridge_regression(r_gen_states_to_use,
                               out_states_to_use,
                               reg_param=reg_param)

st.write(w_out_ridge)

# Adding noise:
st.markdown(
    r"""
    **Adding noise**: 
    """
)
noise_scale = st.number_input("Noise scale", value=0.01, format="%f")
w_out_noise = noisy_linear_regression(r_gen_states_to_use,
                                      out_states_to_use,
                                      noise_scale=noise_scale)

st.write(w_out_noise)

# PCR:
st.markdown(
    r"""
    **Principal component regression**: 
    
    First demonstrate that PCR on non-multicollinear data, with all components is the same 
    as normal linear regression:  
    """
)

st.markdown(
    r"""
    PCR on the non-multicollinear dataset. Show the condition and the rank of the PCA transformed
    moment matrix: $R_\text{pca}^T R_\text{pca}$:
    """
)

# The PCA r_states:

pca_comps = int(st.number_input("PCA components", value=r_dim, min_value=1, max_value=r_dim))


pca = PCA(n_components=pca_comps)
r_states_pca = pca.fit_transform(r_states_to_use)
r_gen_states_pca = get_r_gen_states(r_states_pca)
moment_matrix_pca = get_moment_matrix(r_gen_states_pca)

st.markdown(
    r"""
    Test the condition and rank of the pca moment matrix: 
    """
)
test_for_multicollinearity(moment_matrix_pca)


st.markdown(
    r"""
    Show $W^*_\text{out, pca}$:
    """
)

w_out_pca = linear_regression(r_gen_states_pca, out_states_to_use)

w_out_pca_ridge = ridge_regression(r_gen_states_pca, out_states_to_use, reg_param=reg_param)
w_out_pca = w_out_pca_ridge

st.write(w_out_pca)

st.markdown(
    r"""
    Show $W^*_\text{out}$ backtransformed from $W^*_\text{out, pca}$:
    """
)

pca_component_matrix = pca.components_
w_out_from_pca = transform_pca_w_out_back(w_out_pca, r_states_to_use,
                                          pca_component_matrix)
st.write(w_out_from_pca)


fig = go.Figure()
fig.add_trace(
    go.Bar(x=[f"PC {x+1}" for x in range(pca_comps)], y=pca.explained_variance_)
)
fig.update_xaxes(title="Principal components")
fig.update_yaxes(title="Variance")
fig.update_layout(title="Variance of data points along principle components")
st.plotly_chart(fig)

fig = go.Figure()
fig.add_trace(
    go.Bar(y=np.sum(np.abs(w_out_pca), axis=0))
)
fig.update_xaxes(title="pca_comps + 1")
fig.update_yaxes(title="w_out summed")
fig.update_layout(title="W_out_pca plot")
st.plotly_chart(fig)
