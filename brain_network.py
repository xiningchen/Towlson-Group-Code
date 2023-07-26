"""
Functions for analyzing human brain connectomes. NetworkX is very slow. Code here works with adj. matrix only instead
of networkX graphs so should be much faster on large connectomes. For how to work with NetworkX, see <TBA>.jupyter
notebook demo (coming soon...).

Last update changes:
- Added thresholding code
- Added is_connected code for detecting if a graph is connected
- Updated PC code to work with matrix instead of networkX
- Updated zscore

Future changes:
- Include a jupyter notebook to demo similar functions here but using networkx library instead.

Last updated: July 26, 2023
Author(s): Xining Chen
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import bct
from collections import Counter

WARNING_ON = False


def get_avg_connectome(dir_path, shape, negative_weights=True):
    """
    Get a networkx graph object from a set of connectome data stored in .xlsx type files
    Modify the read_excel() function to read additional rows and columns. Currently set to 94x94 matrix.
    :param dir_path: file path to the folder that contains the connectomes to be averaged
    :param shape: connectome shape
    :return: averaged connectome matrix W (a single 2D array)
    """
    total_connectome = np.zeros(shape=shape)
    num_files = 0
    for root, dirs, files in os.walk(dir_path):
        for file in tqdm(files):
            if not file.endswith('.xlsx'):
                continue
            brain = pd.read_excel(root + file, index_col=0, header=0, nrows=94, usecols="A:CQ")
            brain_numpy_array = brain.to_numpy()
            num_files += 1
            total_connectome = total_connectome + brain_numpy_array
    total_connectome = total_connectome / num_files
    if negative_weights:
        return total_connectome
    else:
        return total_connectome.clip(0)


def apply_threshold(th, W, binarise=False):
    """
    Threshold weighted correlation matrix by retaining x% of the strongest edges.
    :param th: Threshold % (x)
    :param W: Weighted correlation matrix
    :param binarise: If return matrix is a binary matrix of the retained edges or weighted.
    :return: thresholded matrix
    """
    res = np.zeros(W.shape)
    n = W.shape[0]
    ne = int((n * n - n) / 2)
    n_new = int(ne * th - 0.5) + 1
    indices = np.triu_indices(n, k=1)
    edge_weights = W[indices]
    edges = zip(zip(indices[0], indices[1]), edge_weights)
    sorted_edges = sorted(edges, key=lambda x: x[1], reverse=True)
    keep_edges = sorted_edges[:n_new]
    if binarise:
        for ed, _ in keep_edges:
            res[ed[0], ed[1]] = 1
            res[ed[1], ed[0]] = 1
    else:
        for ed, w in keep_edges:
            res[ed[0], ed[1]] = w
            res[ed[1], ed[0]] = w
    return res


def get_degree_distribution(W):
    """
    Returns the degree (or weighted degree if W is weighted adjacency matrix) of each node in the network
    :param W: adjacency matrix (2d numpy array)
    :return: a list of each node's degree
    """
    degree_dist = np.zeros(W.shape[0])
    for i in W.shape[0]:
        degree_dist[i] = np.sum(np.where(W[i] != 0))
    return degree_dist


def get_community_to_node_map(communities, nodeMetaData):
    """
    Converts a partition C = C1, C2, ..., Cn where Ci is the module assignment of node i to a group of node ids belonging
    to the same module assignment.
    :param communities: partition
    :param nodeMetaData:
    :return: A dictionary where the keys are the modules (communities) and the values is a list of node ids that belong
    to that module.
    """
    nodeList = list(nodeMetaData.index)
    community_to_node_list = {c: [] for c in communities.values()}
    for n in nodeList:
        community_to_node_list[communities[nodeMetaData['Functional_System'][n]]].append(n)
    return community_to_node_list


def within_module_zscore(W, partition):
    """
    Calculate the within-module degree z-score for each node in a network.
    :param W: 2D numpy array (connectome) W should be symmetrical (undirected graph) and can be
    weighted or binary. Assumes NO SELF-LOOPS (main diagonal of W is 0).
    :param partition: non-overlapping community affiliation vector (list)
    :return:
    """
    all_modules = set(partition)
    partition_np = np.array(partition)
    within_module_mean = np.zeros(len(partition))
    within_module_std = np.zeros(len(partition))
    within_module_degree = np.zeros(len(partition))
    problematic_nodes = np.full(len(partition), False)
    for m in all_modules:
        within_module_nodes = np.where(partition_np == m)[0]
        dist = np.sum(W[within_module_nodes][:, within_module_nodes], axis=1)
        if np.std(dist) == 0:
            within_module_mean[within_module_nodes] = 0
            within_module_std[within_module_nodes] = 1
            problematic_nodes[within_module_nodes] = True
        else:
            within_module_mean[within_module_nodes] = np.mean(dist)
            within_module_std[within_module_nodes] = np.std(dist)
        within_module_degree[within_module_nodes] = np.sum(W[within_module_nodes][:, within_module_nodes], axis=1)
    zscore = (within_module_degree - within_module_mean) / within_module_std
    zscore[problematic_nodes] = 0
    if WARNING_ON and problematic_nodes.any():
        print(f"WARNING: There are {sum(problematic_nodes)} problematic nodes.")
    return zscore


def participation_coefficient(W, partition):
    """
    Calculate the NORMALIZED participation coefficient for each node in the network. Can be weighted or not weighted.
    :param W: 2D numpy array (connectome) W should be symmetrical (undirected graph) and can be
    weighted or binary. Assumes NO SELF-LOOPS (main diagonal of W is 0).
    :param partition: non-overlapping community affiliation vector (list)
    :return: PC value of each node in W
    """
    if W.shape[0] != W.shape[1]:
        print("ERROR: Bad graph. Exit code.")
        return []
    partition_np = np.array(partition)
    ki_m = np.zeros(W.shape[0])
    for m in set(partition):
        within_module_nodes = np.where(partition_np == m)[0]
        ki_m += np.sum(W[:, within_module_nodes], axis=1) ** 2
    denom = np.sum(W, axis=1) ** 2
    isolated_nodes = np.where(denom == 0)[0]
    if len(isolated_nodes) > 0:
        denom[isolated_nodes] = 1
        pc = np.ones(W.shape[0]) - (ki_m / denom)
        pc[isolated_nodes] = 0
        return pc
    return np.ones(W.shape[0]) - (ki_m / denom)


def get_network_node_roles(W, partition, zscore_threshold=2.5):
    """
    Calculate and assign node roles to a network defined by W.

    Based on the paper "Cartography of complex networks: modules and universal roles" by Guimera and Amaral, 2005.
    Assign node roles to each node of the network. Node roles are defined in the paper mentioned above.
    Hub vs non-hub nodes are determined by the within module zscore. In the paper they used a threshold of > 2.5 being
    hub nodes. This to me was arbitrary, hence this threshold can be toggled as part of the function.
    The node role subcategories are determined by PC values. The boundaries of each role were shown to be
    general (apply to any size of networks, understand Fig. 4 and caption in paper). Hence the PC value boundaries are
    not adjustable in this function.
    :param W: 2D numpy array (connectome) W should be symmetrical (undirected graph) and can be
    weighted or binary. Assumes NO SELF-LOOPS (main diagonal of W is 0).
    :param partition: non-overlapping community affiliation vector (list)
    :param zscore_threshold: determines cut-off for hub nodes
    :return: the nodes' role, zscore, and pc values
    """
    zscore = within_module_zscore(W, partition)
    pc = participation_coefficient(W, partition)
    node_roles = assign_node_roles(zscore, pc, zscore_threshold)
    return node_roles, zscore, pc


def assign_node_roles(zscore, pc, zscore_threshold=2.5):
    """
    Assign node roles to a network defined by W.

    Based on the paper "Cartography of complex networks: modules and universal roles" by Guimera and Amaral, 2005.
    Assign node roles to each node of the network. Node roles are defined in the paper mentioned above.
    Hub vs non-hub nodes are determined by the within module zscore. In the paper they used a threshold of > 2.5 being
    hub nodes. This to me was arbitrary, hence this threshold can be toggled as part of the function.
    The node role subcategories are determined by PC values. The boundaries of each role were shown to be
    general (apply to any size of networks, understand Fig. 4 and caption in paper). Hence the PC value boundaries are
    not adjustable in this function.
    :param zscore:
    :param pc:
    :param zscore_threshold:
    :return:
    """
    if len(zscore) != len(pc):
        print("ERROR: zscore and pc length should be equal (total number of nodes in the network) but they're not?!")
        return []
    roles = [None * len(zscore)]
    for i in range(len(zscore)):
        if zscore[i] >= zscore_threshold:
            # hub node categories
            if pc[i] < 0.3:
                roles[i] = "R5"
            elif pc[i] <0.75:
                roles[i] = "R6"
            elif pc[i] >= 0.75:
                roles[i] = "R7"
            else:
                roles[i] = "Error in PC value"
        else:
            # non-hub categories
            if pc[i] < 0.05:
                roles[i] = "R1"
            elif pc[i] < 0.625:
                roles[i] = "R2"
            elif pc[i] < 0.8:
                roles[i] = "R3"
            elif pc[i] >= 0.8:
                roles[i] = "R4"
            else:
                roles[i] = "Error in PC value"
    return roles


def normalized_participation_coefficient(W, partition, n_iter=100):
    """
    WARNING: The null model code from BCT is *extremely* slow, hence this function is slow. Don't use this.
    Calculate the NORMALIZED participation coefficient for each node in the network. Based on the paper
    "Reducing the influence of intramodular connectivity in participation coefficient" by Pederson et al. 2020.
    Translated code from Github:
    https://github.com/omidvarnia/Dynamic_brain_connectivity_analysis/blob/master/participation_coef_norm.m
    Note: W should only contain positive edges. If the network contains negative edges, need to pre-filter and handle
    positive and negative edges separately.
    :param W: 2D numpy array (weighted adjacency matrix W) W should be symmetrical (undirected graph) and can be
    weighted or binary. Assumes NO SELF-LOOPS (main diagonal of W is 0).
    :param partition: non-overlapping community affiliation vector (list)
    :param n_iter: number of matrix randomizations (default = 100)
    :return: PC value of each node in W
    """
    if W.shape[0] != W.shape[1]:
        print("ERROR: Bad graph. Exit code.")
        return []
    n = W.shape[0]
    modules = set(partition)
    M = len(modules)
    partition_np = np.array(partition)
    ki = np.sum(W, axis=1) ** 2
    isolated_nodes = np.where(ki == 0)[0]
    if len(isolated_nodes) > 0:
        ki[isolated_nodes] = 1
    ki_m = np.zeros((n, M))
    within_module_nodes = {m: None for m in modules}
    for i, m in enumerate(modules):
        within_module_nodes[m] = np.where(partition_np == m)[0]
        ki_m[:, i] = np.sum(W[:, within_module_nodes[m]], axis=1)
    rnd_samples = np.zeros((n, n_iter))
    for i in tqdm(range(n_iter)):
        W_rnd = bct.null_model_und_sign(W, 5)
        print("Created a null model...")
        x = np.zeros(n)
        for m in modules:
            ki_m_rnd = np.sum(W_rnd[:, within_module_nodes[m]], axis=1)
            x += (ki_m[:, m] - ki_m_rnd) ** 2
        rnd_samples[:, i] = math.sqrt(0.5 * (x / ki))
    pc_norm = np.ones(n) - np.median(rnd_samples, axis=1)
    if WARNING_ON and len(isolated_nodes) > 0:
        print("There are isolated nodes. Setting their PC to be 0.")
        pc_norm[isolated_nodes] = 0
    return pc_norm



def flexibility(cycle_partitions):
    """
    Nodal flexibility is defined as the number of times a node changes in community allegiance across network layers,
    normalized by the maximum number of possible changes.
    :param cycle_partitions: list of partitions (list). Note, this list should be ordered from time = 0 to time = l
    :return: nodal flexibility of each node in a network
    """
    partis = cycle_partitions.copy()
    partis.append(partis[0])
    max_switches = len(partis) - 1
    f_i = np.zeros(len(partis[0]))
    for i in range(max_switches):
        cur_p = partis[i]
        next_p = partis[i + 1]
        switched = np.array(cur_p) != np.array(next_p)
        f_i += switched
    return f_i / max_switches


def autocorrelation_fct(t, tm, icn_i):
    """
    Defined as the number of common nodes found in group $i$ at time $t$ and $t+m$, normalized by the combined number of
    nodes found in group $i$ at both time points.
    :param t: partition at time t
    :param tm: partition at time t+m
    :param icn_i: group $i$
    :return:
    """
    t_set = set([j for j, ni in enumerate(t) if ni == icn_i])
    tm_set = set([j for j, ni in enumerate(tm) if ni == icn_i])
    return len(t_set.intersection(tm_set)) / len(t_set.union(tm_set))


def nodal_association_matrix(partition):
    """
    The boolean matrix T where T_{ij} is 1 if times nodes $i$ and $j$ are found in the same community, 0 otherwise.
    The sum of this matrix over a set of partitions is called the module allegiance matrix.
    :param partition: non-overlapping community affiliation vector (list)
    :return: T
    """
    T = np.eye(1054)
    for i in range(1054):
        for j in range(i + 1, 1054):
            if partition[i] == partition[j]:
                T[i][j] += 1
                T[j][i] += 1
    return T


def interaction_strength(partition, maP, k1, k2):
    """
    Interaction strength.
    :param partition: non-overlapping community affiliation vector (list)
    :param maP: module allegiance matrix
    :param k1: group 1 (module 1)
    :param k2: group 2 (module 2)
    :return:
    """
    L = len(partition)
    size_k1 = Counter(partition)[k1]
    size_k2 = Counter(partition)[k2]
    if (size_k1 == 0) or (size_k2 == 0):
        return 0
    P_sum = 0
    for i in range(L):
        if partition[i] != k1:
            continue
        for j in range(L):
            if partition[j] != k2:
                continue
            P_sum += maP[i][j]
    return P_sum / (size_k1 * size_k2)


def average_recruitment(partition, maP, k1):
    """
    Average recruitment is the within module interaction strength
    :param partition: non-overlapping community affiliation vector (list)
    :param maP: module allegiance matrix
    :param k1: group (module)
    :return:
    """
    return interaction_strength(partition, maP, k1, k1)


def average_integration(partition, maP, k1, k2):
    """
    Average integration is the between module interaction strength
    :param partition: non-overlapping community affiliation vector (list)
    :param maP: module allegiance matrix
    :param k1: group 1 (module)
    :param k2: group 2 (module)
    :return:
    """
    return interaction_strength(partition, maP, k1, k2)


def relative_interaction_strength(partition, maP, k1, k2):
    """
    Relative interaction strength is normalizing interaction strength with group's internal interaction strength.
    :param partition: non-overlapping community affiliation vector (list)
    :param maP: module allegiance matrix
    :param k1: group 1 (module 1)
    :param k2: group 2 (module 2)
    :return:
    """
    if k1 == k2:
        print("Error: Integration is between two different groups (k1 != k2).")
        return -1
    norm_int = interaction_strength(partition, maP, k1, k2)
    I_k1 = interaction_strength(partition, maP, k1, k1)
    I_k2 = interaction_strength(partition, maP, k2, k2)
    if (I_k1 == 0) or (I_k2 == 0):
        # This means one of these modules don't exist, hence no interaction strength with itself
        return 0
    return norm_int / (math.sqrt(I_k1 * I_k2))


def connectivity_strength(partition, W, k1, k2=None):
    """
    Measures the connectivity strength within or between modules. If k2 is None, then this calculates intraconnectivity
    strength. Else it calculates interconnectivity strength. Note that this is *not* the average edge weight
    within/between modules. Strength is a measure of the amount of connection w.r.t. maximum possible connections. If
    group 1 has N1 nodes and group 2 has N2 nodes, then the maximum number of possible edges is N1*N2.
    When k2 is None, calculate within module strength of module k1. Else, calculate between module strength between
    modules k1 and k2. For weighted functional brain networks, negative edges could mean something completely
    different, hence treat positive and negative edges separately.
    Warning: W must be a symmetric matrix.
    :param partition: non-overlapping community affiliation vector (list)
    :param W: numpy array for a *symmetric* weighted adjacency matrix
    :param k1: module/group 1 (defined in the partition)
    :param k2: (optional) module/group 2 (defined in the partition)
    :return:
    """
    pos_W = W * (W >= 0)
    neg_W = W * (W <= 0)
    if k2 is None:
        k2 = k1
    k1_nodes = [i for i, v in enumerate(partition) if v == k1]
    k2_nodes = [i for i, v in enumerate(partition) if v == k2]
    if len(k1_nodes) == 0 or len(k2_nodes) == 0:
        return 0, 0
    if len(k1_nodes) <= len(k2_nodes):
        small_set = k1_nodes
        large_set = k2_nodes
    else:
        small_set = k2_nodes
        large_set = k1_nodes
    pos_edges_w = 0
    neg_edges_w = 0
    for n1 in small_set:
        # Positive
        between_edges = pos_W[n1, large_set]
        pos_edges_w += sum(between_edges)
        # Negative
        between_edges = neg_W[n1, large_set]
        neg_edges_w += sum(between_edges)
    pos_con = pos_edges_w / (len(k1_nodes) * len(k2_nodes))
    neg_con = neg_edges_w / (len(k1_nodes) * len(k2_nodes))
    return pos_con, neg_con


def nodal_connectivity_strength(W, k1, k2=None):
    """
    This is different from connectivity strength function only by the input parameters. Otherwise it is the same calculation.
    :param W: numpy array for a *symmetric* weighted adjacency matrix
    :param k1: group 1 list of nodes
    :param k2: (optional) group 2 list of nodes
    :return:
    """
    pos_W = W * (W >= 0)
    neg_W = W * (W <= 0)
    if k2 is None:
        k2_nodes = k1
    else:
        k2_nodes = k2
    k1_nodes = k1
    if len(k1_nodes) == 0 or len(k2_nodes) == 0:
        return 0, 0
    if len(k1_nodes) <= len(k2_nodes):
        small_set = k1_nodes
        large_set = k2_nodes
    else:
        small_set = k2_nodes
        large_set = k1_nodes
    pos_edges_w = 0
    neg_edges_w = 0
    for n1 in small_set:
        # Positive
        between_edges = pos_W[n1, large_set]
        pos_edges_w += sum(between_edges)
        # Negative
        between_edges = neg_W[n1, large_set]
        neg_edges_w += sum(between_edges)
    pos_con = pos_edges_w / (len(k1_nodes) * len(k2_nodes))
    neg_con = neg_edges_w / (len(k1_nodes) * len(k2_nodes))
    return pos_con, neg_con


def adj_mat_grouped_by_modules(W, partition):
    """
    Given a matrix, shuffle rows/columns so that nodes belonging to the same community are grouped together.
    :param W: matrix to shuffle (network)
    :param partition: non-overlapping community affiliation vector (list)
    :return: a shuffled matrix W' grouped based on the partition information
    """
    N = W.shape[0]
    shuf = sorted(zip(range(0, N), partition), key=lambda x: x[1])
    shuf_order, _ = zip(*shuf)
    plot_mat = np.zeros((N, N))
    for i in range(N):
        r = shuf_order[i]
        ri = np.zeros(1054)
        for s, j in enumerate(shuf_order):
            ri[s] = W[r][j]
        plot_mat[i, :] = ri
        plot_mat[:, i] = ri
    return plot_mat


def is_connected(A):
    """
    Calculate the connected components of a graph defined by its adjacency matrix A.
    :param A: Numpy 2D array - the adjacency (boolean) matrix of a graph
    :return: number of components (int), list of components (list of lists)
    """
    adj_list = []
    for i in range(A.shape[0]):
        adj_list.append(list(np.nonzero(A[i, :])[0]))
    node_list = np.zeros(A.shape[0])
    num_connected_components = 0
    connected_components = []
    while len(np.where(node_list == 0)[0]) > 0:
        root = np.where(node_list == 0)[0][0]
        num_connected_components += 1
        a_component = [root]
        stack = adj_list[root]
        node_list[root] = 1
        # DFS
        while len(stack) > 0:
            next = stack.pop()
            if node_list[next] == 1:
                continue
            node_list[next] = 1
            a_component.append(next)
            stack = stack + adj_list[next]
        connected_components.append(a_component)
    return num_connected_components, connected_components


# ------------------------------------------------------------------------------------------------------------------------

def get_network_diagnostic(partitions_list):
    """
    Based on "Robust detection of dynamic community structure in networks" (Bassett2013a), the 4 network diagnostic
    measures are:
    1. Modularity: single-layer modularity value for each partition
    2. Number of communities: number of communities in each partition layer
    3. Community size: community size for each layer
    4. Stationarity: stationarity of each community
    :param partitions_list: a list of partitions. This should be grouped appropriately.
    :return: dictionary containing the network diagnostic values
    """
    mean_community_size = []
    for a_part in partitions_list:
        ccc = Counter(a_part)
    res = {'Modularity': 0, 'Number of communities': len(ccc), 'Community size': 0, 'Stationarity': 0}

# def get_network_stats(G):
#     """
#     Given a network G, print basic network statistics.
#     Network stats:
#     (1) Average weight (if weighted) (float)
#     (2) Average degree (float)
#     (3) Degree distribution
#     (4) Weight distribution (if weighted) (list)
#     (5) Average path length (float)
#     (6) Average clustering coefficient (float)
#     :param G: NetworkX graph G (can be weighted or unweighted)
#     :return: dictionary containing network information
#     """
#     result = {'avg_degree': 0.0,
#               'avg_weight': 0.0,
#               'degree_seq': [],
#               'weight_seq': [],
#               'avg_path_length': 0.0,
#               'avg_clustering_coef': 0.0}
#     paths = []
#     for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
#         paths.append(nx.average_shortest_path_length(C))
#     result['avg_path_length'] = float(sum(paths)) / len(paths)
#     result['avg_clustering_coef'] = nx.average_clustering(G)
#     # Degree distribution
#     degree_seq = sorted([d for n, d in G.degree()], reverse=True)
#     result['avg_degree'] = sum(degree_seq) / len(degree_seq)
#     result['degree_seq'] = degree_seq
#     # Weighted distribution
#     attributes = nx.get_edge_attributes(G, 'weight')
#     if len(attributes) != 0:
#         weight_seq = sorted([G.degree(node, weight="weight") for node in G], reverse=True)
#         result['avg_weight'] = sum(weight_seq) / len(weight_seq)
#         result['weight_seq'] = weight_seq
#     return result


# def get_network_stats2(G):
#     """
#     More basic network metrics.
#     :param G:
#     :return:
#     """
#     result = {'avg_path_length': nx.average_shortest_path_length(G),
#               'clustering': np.mean(list(nx.clustering(G, weight="weight").values())),
#               'shortest_path': nx.average_shortest_path_length(G),
#               'global_efficiency': nx.global_efficiency(G)}
#     node_strengths = sorted([(node, G.degree(node, weight="weight")) for node in G.nodes()], reverse=True,
#                             key=lambda x: x[1])
#     avg_node_strength = [x[1] for x in node_strengths]
#     result['avg_weight'] = sum(avg_node_strength) / len(avg_node_strength)
#     return result
