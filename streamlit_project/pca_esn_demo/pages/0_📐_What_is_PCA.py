"""A streamlit app to demonstrate PCA is conjunction with Echo State Networks - Linear Reg."""

import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import streamlit as st


st.set_page_config("What is PCA", page_icon="📐")

st.header("What is PCA?")

st.markdown(
    """
    The Principal Component Analysis is a technique that can be applied to a collection of points, 
    to find directions within the space of points, which **show the largest variance in the 
    data**. Mathematically, PCA is an orthogonal linear transformation to a new coordinate system. 
   
    **PCA works in two steps:** 
    - Shift the origin of the new coordinate system to the mean of the data points. 
    - The directions of the new coordinate system axes (**principal components**) are defined as:
        - The **first principal component** is the direction that maximizes the variance of the 
            data points, when projected onto that axis. 
        - The **second principal component** is the direction that is orthogonal to the first PC 
            **and** maximizes the variance of the data points, and so on. 
        - The **n'th PC** must be orthogonal to all previous PCs and then maximize the variance 
            within the data.  
    
    **Mathematically, the PCA transforms the original datapoints into the pca-coordinate system:**
    """
)

st.latex(
    r"""
    \begin{aligned}
    \boldsymbol{r}_\text{pca} &= P (\boldsymbol{r} - \boldsymbol{r}_\text{mean}) \\
    \end{aligned}
    """
)

st.subheader("Demonstration: ")
st.markdown("""
Demonstrate PCA on a collection of 2-dimensional data points, drawn from a 
multivariate normal distribution with correlations.
""")

# Easy PCA example:
nr_pnts = 1000
data_mean = np.array([5, 5])
data_cov = np.array([[2, 1], [1, 1]])

seed = 0
rng = np.random.default_rng(seed=seed)
data_matrix = rng.multivariate_normal(mean=data_mean, cov=data_cov, size=nr_pnts)

pca = PCA()
data_matrix_transformed = pca.fit_transform(data_matrix)
pca_components = pca.components_
pca_explained_var = pca.explained_variance_
data_mean = np.mean(data_matrix, axis=0)

max_x = np.max(data_matrix[0, :])
max_y = np.max(data_matrix[1, :])

# Plot 1: PCA components ontop of 2d data cloud.
# draw data points:
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=data_matrix[:, 0],
               y=data_matrix[:, 1],
               name="Data points",
               mode='markers'),
)

# # Draw mean of data points:
# fig.add_trace(
#     go.Scatter(x=[0, data_mean[0]],
#                y=[0, data_mean[1]],
#                name="Mean of data points")
# )

# Draw first pca component:
fig.add_trace(
    go.Scatter(x=[data_mean[0], data_mean[0] + pca_components[0, 0]],
               y=[data_mean[1], data_mean[1] + pca_components[0, 1]],
               name="PCA component 1")
)
# Draw second pca component:
fig.add_trace(
    go.Scatter(x=[data_mean[0], data_mean[0] + pca_components[1, 0]],
               y=[data_mean[1], data_mean[1] + pca_components[1, 1]],
               name="PCA component 1")
)

fig.update_xaxes(range=[0, 10],
                 title="Data dimension 0")
fig.update_yaxes(range=[0, 10],
                 scaleanchor="x",
                 scaleratio=1,
                 title="Data dimension 1")
fig.update_layout(title="Original data points (r) with principal components:", )
st.plotly_chart(fig)

# Plot 2:
# draw data points in pca coordinate system:
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=data_matrix_transformed[:, 0],
               y=data_matrix_transformed[:, 1],
               name="PCA transformed data points",
               mode='markers'),
)

# Draw first pca component:
fig.add_trace(
    go.Scatter(x=[0, 1],
               y=[0, 0],
               name="PCA component 1")
)
# Draw second pca component:
fig.add_trace(
    go.Scatter(x=[0, 0],
               y=[0, 1],
               name="PCA component 2")
)

fig.update_xaxes(
                 title="pca component 0")
fig.update_yaxes(
                 scaleanchor="x",
                 scaleratio=1,
                 title="pca component 1")

fig.update_layout(title="Data points after pca transformation (r_pca):")
st.plotly_chart(fig)


# Plot 3: Explained variance of the PCA components.
fig = go.Figure()
fig.add_trace(
    go.Bar(x=["PC 1", "PC 2"], y=pca_explained_var)
)
fig.update_xaxes(title="Principal components")
fig.update_yaxes(title="Variance")
fig.update_layout(title="Variance of data points along principle components")
st.plotly_chart(fig)
