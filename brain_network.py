"""
Functions for creating network x graphs using brain connectome data and computing a few network metrics given a
network.

Last updated: April 24, 2023
Author(s): Xining Chen
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import statistics as stats
import math
from collections import Counter


def get_avg_connectome(dir_path, shape, negative_weights=True):
    """
    Get a networkx graph object from a set of connectome data stored in .xlsx type files
    Modify the read_excel() function to read additional rows and columns. Currently set to 94x94 matrix.
    :param dir_path: file path to the folder that contains the connectomes to be averaged
    :param shape: connectome shape
    :return: networkx graph with edge weights
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
        avg_nx_graph = nx.from_numpy_array(total_connectome)
        avg_nx_graph.edges(data=True)
    else:
        avg_nx_graph = nx.from_numpy_array(total_connectome.clip(0))
        avg_nx_graph.edges(data=True)
    return avg_nx_graph


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
    ne = int((n*n - n)/2)
    n_new = int(ne*th - 0.5) + 1
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

def get_network_stats(G):
    """
    Given a network G, print basic network statistics.
    Network stats:
    (1) Average weight (if weighted) (float)
    (2) Average degree (float)
    (3) Degree distribution
    (4) Weight distribution (if weighted) (list)
    (5) Average path length (float)
    (6) Average clustering coefficient (float)
    :param G: NetworkX graph G (can be weighted or unweighted)
    :return: dictionary containing network information
    """
    result = {'avg_degree': 0.0,
              'avg_weight': 0.0,
              'degree_seq': [],
              'weight_seq': [],
              'avg_path_length': 0.0,
              'avg_clustering_coef': 0.0}
    paths = []
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        paths.append(nx.average_shortest_path_length(C))
    result['avg_path_length'] = float(sum(paths)) / len(paths)
    result['avg_clustering_coef'] = nx.average_clustering(G)
    # Degree distribution
    degree_seq = sorted([d for n, d in G.degree()], reverse=True)
    result['avg_degree'] = sum(degree_seq) / len(degree_seq)
    result['degree_seq'] = degree_seq
    # Weighted distribution
    attributes = nx.get_edge_attributes(G, 'weight')
    if len(attributes) != 0:
        weight_seq = sorted([G.degree(node, weight="weight") for node in G], reverse=True)
        result['avg_weight'] = sum(weight_seq) / len(weight_seq)
        result['weight_seq'] = weight_seq
    return result


def get_network_stats2(G):
    """
    More basic network metrics.
    :param G:
    :return:
    """
    result = {'avg_path_length': nx.average_shortest_path_length(G),
              'clustering': np.mean(list(nx.clustering(G, weight="weight").values())),
              'shortest_path': nx.average_shortest_path_length(G),
              'global_efficiency': nx.global_efficiency(G)}
    node_strengths = sorted([(node, G.degree(node, weight="weight")) for node in G.nodes()], reverse=True,
                            key=lambda x: x[1])
    avg_node_strength = [x[1] for x in node_strengths]
    result['avg_weight'] = sum(avg_node_strength) / len(avg_node_strength)
    return result


def get_degree_distribution(W, weighted=False):
    # For adj. matrix W
    np_W = np.array(W)
    zero_W = (np_W != 0)

    # For networkX graph


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


def get_zscore(G, communities, community_to_node_list):
    """
    Calculate weighted z-score for each node in graph G.
    :param G:
    :param communities:
    :param community_to_node_list:
    :return: z-score for each community, graph G with z-score attribute added to each node.
    """
    zscore_per_community = {c: [] for c in communities.values()}
    zscore_attrs = {n: {"zscore": 0} for n in G.nodes()}
    for comNum, nlist in community_to_node_list.items():
        w = []
        for v1 in nlist:
            w_i = 0
            for v2 in nlist:
                if not (v2 in G[v1]):
                    continue
                w_i += G[v1][v2]["weight"]
            w.append(w_i)
        data = np.array(w)
        zscore_per_community[comNum] = stats.zscore(data)
        for i, v1 in enumerate(nlist):
            zscore_attrs[v1]["zscore"] = zscore_per_community[comNum][i]
        nx.set_node_attributes(G, zscore_attrs)
        # print(zscore_per_community[comNum])
    return zscore_per_community, G


def get_pc(G, nodeList, community_to_node_list):
    """
    Calculate weighted participation coefficient for each node in graph G.
    :param G:
    :param nodeList:
    :param community_to_node_list:
    :return:
    """
    pc = {n: 0 for n in nodeList}
    pc_attrs = {n: {"pc": 0} for n in G.nodes()}
    for node in nodeList:
        w_i = G.degree(node, weight="weight")
        tot = 0
        for comNum, nlist in community_to_node_list.items():
            w_is = 0
            for n in nlist:
                if not (n in G[node]):
                    continue
                w_is += G[node][n]["weight"]
            tot += (w_is / w_i) ** 2
        if (tot > 1):
            print(f"ERROR - {node}")
            break
        pc[node] = 1 - tot
        pc_attrs[node]["pc"] = 1 - tot
    nx.set_node_attributes(G, pc_attrs)
    return pc, G


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


def average_flexibility(cycle_partitions, icn_i=-1, nodal=False):
    """
    The average flexibility of a module/network is the average of all nodal flexibility in that module.
    :param cycle_partitions: list of partitions (list). Note, this list should be ordered from time = 0 to time = l
    :param icn_i: module id that corresponds to the module labels in the partitions, default = -1 which will get flexibility
    of all nodes.
    :param nodal: Boolean, whether to return node level flexibility or not.
    :return: average flexibility of a module/group/network, node level flexibility
    """

    N = len(cycle_partitions[0])
    if icn_i != -1:
        noi_is = [j for j, ni in enumerate(cycle_partitions[0]) if ni == icn_i]
        if len(noi_is) == 0:
            return 0, []
        f_is = flexibility(cycle_partitions)
        avg_flex = 0
        node_flex = [0] * N
        for noi in noi_is:
            if nodal:
                node_flex[noi] = f_is[noi]
            avg_flex += f_is[noi]
        return avg_flex / len(noi_is), node_flex
    else:
        f_is = flexibility(cycle_partitions)
        avg_flex = sum(f_is) / len(f_is)
        if nodal:
            return avg_flex, f_is
        else:
            return avg_flex, []


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


def nodal_association_matrix(a_part):
    """
    The boolean matrix T where T_{ij} is 1 if times nodes $i$ and $j$ are found in the same community, 0 otherwise.
    The sum of this matrix over a set of partitions is called the module allegiance matrix.
    :param a_part: a single partition
    :return: T
    """
    T = np.eye(1054)
    for i in range(1054):
        for j in range(i + 1, 1054):
            if a_part[i] == a_part[j]:
                T[i][j] += 1
                T[j][i] += 1
    return T


def interaction_strength(a_part, maP, k1, k2):
    """
    Interaction strength.
    :param a_part: a partition
    :param maP: module allegiance matrix
    :param k1: group 1 (module 1)
    :param k2: group 2 (module 2)
    :return:
    """
    L = len(a_part)
    size_k1 = Counter(a_part)[k1]
    size_k2 = Counter(a_part)[k2]
    if (size_k1 == 0) or (size_k2 == 0):
        return 0
    P_sum = 0
    for i in range(L):
        if a_part[i] != k1:
            continue
        for j in range(L):
            if a_part[j] != k2:
                continue
            P_sum += maP[i][j]
    return P_sum / (size_k1 * size_k2)


def average_recruitment(a_part, maP, k1):
    """
    Average recruitment is the within module interaction strength
    :param a_part: a partition
    :param maP: module allegiance matrix
    :param k1: group (module)
    :return:
    """
    return interaction_strength(a_part, maP, k1, k1)


def average_integration(a_part, maP, k1, k2):
    """
    Average integration is the between module interaction strength
    :param a_part: a partition
    :param maP: module allegiance matrix
    :param k1: group 1 (module)
    :param k2: group 2 (module)
    :return:
    """
    return interaction_strength(a_part, maP, k1, k2)


def relative_interaction_strength(a_part, maP, k1, k2):
    """
    Relative interaction strength is normalizing interaction strength with group's internal interaction strength.
    :param a_part: a partition
    :param maP: module allegiance matrix
    :param k1: group 1 (module 1)
    :param k2: group 2 (module 2)
    :return:
    """
    if k1 == k2:
        print("Error: Integration is between two different groups (k1 != k2).")
        return -1
    norm_int = interaction_strength(a_part, maP, k1, k2)
    I_k1 = interaction_strength(a_part, maP, k1, k1)
    I_k2 = interaction_strength(a_part, maP, k2, k2)
    if (I_k1 == 0) or (I_k2 == 0):
        # This means one of these modules don't exist, hence no interaction strength with itself
        return 0
    return norm_int / (math.sqrt(I_k1 * I_k2))


def connectivity_strength(a_part, W, k1, k2=None):
    """
    Measures the connectivity strength within or between modules. If k2 is None, then this calculates intraconnectivity
    strength. Else it calculates interconnectivity strength. Note that this is *not* the average edge weight
    within/between modules. Strength is a measure of the amount of connection w.r.t. maximum possible connections. If
    group 1 has N1 nodes and group 2 has N2 nodes, then the maximum number of possible edges is N1*N2.
    When k2 is None, calculate within module strength of module k1. Else, calculate between module strength between
    modules k1 and k2. For weighted functional brain networks, negative edges could mean something completely
    different, hence treat positive and negative edges separately.
    Warning: W must be a symmetric matrix.
    :param a_part: partition
    :param W: numpy array for a *symmetric* weighted adjacency matrix
    :param k1: module/group 1 (defined in the partition)
    :param k2: (optional) module/group 2 (defined in the partition)
    :return:
    """
    pos_W = W * (W >= 0)
    neg_W = W * (W <= 0)
    if k2 is None:
        k2 = k1
    k1_nodes = [i for i, v in enumerate(a_part) if v == k1]
    k2_nodes = [i for i, v in enumerate(a_part) if v == k2]
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
    * This is different from connectivity strength function only by the input parameters. Otherwise it is the same calculation.
    Measures the connectivity strength within or between sets of nodes. If k2 is None, then this calculates intraconnectivity
    strength. Else it calculates interconnectivity strength. Note that this is *not* the average edge weight
    within/between modules. For weighted functional brain networks, negative edges could mean something completely
    different, hence treat positive and negative edges separately.
    Warning: W must be a symmetric matrix.
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

# ------------------------------------------------------------------------------------------------------------------------
def stationarity(partitions_list):
    """

    :param partitions_list:
    :return:
    """


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
