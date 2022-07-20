import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import rescomp.esn_new_update_code as esn_new

x_dim = 3
r_dim = 500

n_rad = 0.1
n_avg_deg = 20


# Use the esn create network methods.
# n_type_opt = "erdos_renyi"
# n_type_opt = "erdos_renyi_directed"
# esn = esn_new.ESN_normal()
# esn.build(x_dim, r_dim=r_dim, n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt)
# network = esn.return_network()
# no_self_connections = np.all(np.diag(network) == 0)
# print("")


# directed vs undirected graph
n_edge_prob = n_avg_deg / (r_dim - 1)


# network_nondir = nx.fast_gnp_random_graph(r_dim, n_edge_prob,
#                                seed=np.random)
# degree_nondir = network_nondir.degree
# avg_deg_nondir = np.mean([x[1] for x in degree_nondir])
# adjecency_nondir = nx.to_numpy_array(network_nondir)  # .toarray()
# avg_deg_nondir_alt = np.mean(np.sum(adjecency_nondir, axis=1))
#
# network_dir = nx.fast_gnp_random_graph(r_dim, n_edge_prob,
#                                    seed=np.random, directed=True)
# degree_dir = network_dir.degree
# avg_deg_dir = np.mean([x[1] for x in degree_dir])
# adjecency_dir = nx.to_numpy_array(network_dir)  # .toarray()
#
# avg_deg_dir_alt = np.mean(np.sum(adjecency_dir, axis=1))


# scipy sparse matrix:
network_scipy = scipy.sparse.random(r_dim, r_dim, density=n_edge_prob, random_state=1).toarray()

plt.hist(network_scipy[network_scipy != 0]); plt.show()

nw_scipy_ones = network_scipy.copy()
nw_scipy_ones[nw_scipy_ones != 0] = 1

avg_deg_scipy = np.mean([x[1] for x in nw_scipy_ones])

print("")

