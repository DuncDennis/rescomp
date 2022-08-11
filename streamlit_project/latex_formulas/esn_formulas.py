"""Python file to get latex equation strings for all esn purposes."""

# TODO: clean up and add more.

general_update_equation = r"""r_{i+1} = f\left( g_\text{inp}(x_i) + g_\text{int}(r_i) + 
\text{bias} \right)"""

w_in_and_network_update_equation = r"""r_{i+1} = f\left(W_\text{in} \cdot x_i + W_\text{network} 
\cdot r_i + \text{bias} \right)"""

w_in_and_network_update_equation_with_explanation = r"""
\overbrace{r_{i+1}}^\text{Res. states} = 
f(
\underbrace{
\overbrace{W_\text{in} \cdot x_i}^\text{Res. input} + 
\overbrace{W_\text{network} \cdot r_i}^\text{Res. internal update} + 
\text{bias} 
}_\text{Act. fct. argument}
)"""

w_out_sum_over_r_gen_left = r"""
a_i = \sum_{j=0} |W_\text{out}(i, j)|
"""

w_out_sum_over_r_gen_right = r"""
\begin{aligned}
& i: \text{Output index} \\
& j: \text{R generalized index}\\
\end{aligned}
"""
