"""Python file for esn pca latex formulas."""

pca_transformation = r"""

\boldsymbol{p} = M (\boldsymbol{r} - \boldsymbol{r}_\text{mean}) = \boldsymbol{p} = M \boldsymbol{d}
"""
#
# pca_transformation_definition = r"""
# \boldsymbol{p}: \text{PCA vector}\\
# M: \text{PCA component matrix}\\
#
# """

pca_transformation_definition = r"""
\begin{aligned}
& \boldsymbol{p}: \text{PCA transformed vector} \\
& M: \text{PCA component matrix}\\
& \boldsymbol{r}: \text{reservoir state}\\
& \boldsymbol{r}_\text{mean} = \langle \boldsymbol{r} \rangle: \text{mean of reservoir state}\\
& \boldsymbol{d} = \boldsymbol{r} - \boldsymbol{r}_\text{mean}: \text{res state from mean POV}
\end{aligned}
"""

reservoir_states = r"""
\boldsymbol{r} = \sum_i  r_i \times \hat{\boldsymbol{r}}_i = [r_1, r_2, r_3, ...]^\intercal
"""

# pca_components = r"""
# \hat{\boldsymbol{m}}_j = \sum_i M_{i, j} \times \hat{\boldsymbol{r}}_i = [M_{1, j}, M_{2, j}, ...]^\intercal
# """

reservoir_as_function_of_input_approx = r"""
\boldsymbol{r} = \tanh\left( W_\text{in} \boldsymbol{x} + W\boldsymbol{r}_\text{previous} + 
\text{bias} \right)\approx W_\text{in} \boldsymbol{x}
"""

pca_vector_as_fct_of_input = r"""
\boldsymbol{p} \approx M(W_\text{in} \boldsymbol{x} - \boldsymbol{r}_\text{mean})
"""

input_states = r"""
\boldsymbol{x} = \sum_i  x_i \times \hat{\boldsymbol{x}}_i = [x_1, x_2, ...]^\intercal
"""

pca_components_as_fct_of_input = r"""
\boldsymbol{n}_j = \sum_i [M\times W_\text{in}]^\intercal_{i, j} \times \hat{\boldsymbol{x}}_i = [[M\times W_\text{in}]_{j, 1}, [M\times W_\text{in}]_{j, 2}, ...]
"""

pca_inverse_transformation = r"""
\boldsymbol{d} = M^{-1}\boldsymbol{p} = M^\intercal\boldsymbol{p}
"""

pca_components = r"""
\boldsymbol{m}_j = M^\intercal\hat{\boldsymbol{p}}_j = \sum_i M^\intercal_{i, j} \hat{\boldsymbol{r}}_i = [M_{j, 1}, M_{j, 2}, ...]^\intercal\\

\begin{aligned}
&  \hat{\boldsymbol{p}}_j = [0, ..., 1, ..., 0]^\intercal: \text{pca component unit vector}\\
& \boldsymbol{m}_j: \text{pca component as a combination of reservoir unit vectors}
\end{aligned}
"""

res_state_unit_vectors_explanation = r"""
\begin{aligned}
&  \hat{\boldsymbol{r}}_i = \hat{\boldsymbol{d}}_i = [0, ..., 1, ..., 0]^\intercal: \text{reservoir state unit vectors}\\
\end{aligned}

"""
