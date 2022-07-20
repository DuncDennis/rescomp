"""Comparing different network implementations, via networkx and scipy.sparse.

Aim is to understand how one can implement a "low degree weighted (un)directed Erdös-Renyi network"
using scipy of networkx.
"""

import numpy as np
import scipy
import scipy.sparse as sparse
import networkx as nx
import matplotlib.pyplot as plt


def check_network(nw):
    nw_nx = nx.from_numpy_matrix(nw, create_using=nx.DiGraph)

    try:
        indeg = np.mean([x[1] for x in nw_nx.in_degree])
        print(f"NX: Avg in degree: {indeg}")
    except:
        pass

    try:
        outdeg = np.mean([x[1] for x in nw_nx.out_degree])
        print(f"NX: Avg out degree: {outdeg}")
    except:
        pass
    try:
        print(f"NX: In + Out : {indeg + outdeg}")
    except:
        pass

    print(f"NX: Avg degree: {np.mean([x[1] for x in nw_nx.degree])}")

    print(f"Avg x_sum degree: {np.mean(np.sum(nw, axis=0))}")
    print(f"Avg y_sum degree: {np.mean(np.sum(nw, axis=1))}")

    print(f"Diag zero free: {np.all(np.diag(nw) == 0)}")

    print(f"std x_sum degree: {np.std(np.sum(nw, axis=0))}")
    print(f"std y_sum degree: {np.std(np.sum(nw, axis=1))}")

r_dim = 500

# Question one: What is the average degree of a "directed" network:
print("Q1: What is the average degree of a directed network:")
avg_deg = 100
p = avg_deg/(r_dim-1)

nw_x_dir = nx.fast_gnp_random_graph(r_dim, p=p, seed=np.random, directed=True)
nw_dir = nx.to_numpy_array(nw_x_dir)
check_network(nw_dir)


# Question two: What is the difference between the directed=True and directed=False option in
# nx.ast_gnp_random_graph?
print("_________")
print("Q2:  What is the difference between the directed=True and directed=False option in "
      "nx.ast_gnp_random_graph? ")
avg_deg = 100
p = avg_deg/(r_dim-1)
nw_x_dir = nx.fast_gnp_random_graph(r_dim, p=p, seed=np.random, directed=True)
nw_x_undir = nx.fast_gnp_random_graph(r_dim, p=p, seed=np.random, directed=False)
nw_dir = nx.to_numpy_array(nw_x_dir)
nw_undir = nx.to_numpy_array(nw_x_undir)

print("directed: ")
check_network(nw_dir)
print("undirected: ")
check_network(nw_undir)


# Question three: How can one create a erdos-renyi network with scipy?
print("_________")
print("Q3: How can one create a erdos-renyi with scipy.sparse? (like Pathak)")
# for degree see: Pathak: https://github.com/pvlachas/RNN-RC-Chaos/blob/a403e0e843cf9dde11833f0206f94f91169a4661/Methods/Models/esn/esn.py#L166
avg_deg = 3
density = avg_deg/r_dim
nw_scipy_dir = sparse.random(r_dim, r_dim, density=density).toarray()
nw_scipy_dir_ones = nw_scipy_dir.copy()
nw_scipy_dir_ones[nw_scipy_dir_ones != 0] = 1
print("nw_scipy_dir: ")
check_network(nw_scipy_dir)
print("nw_scipy_dir_ones: ")
check_network(nw_scipy_dir_ones)


