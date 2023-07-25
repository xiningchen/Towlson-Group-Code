"""
Functions for creating partitions (community detection) for functional brain connectomes.
Also includes two partition comparison metrics used in literature:
- Normalized Mutual Information (NMI)
- Variation of Information (VI)
These two are effectively the same, just inverse of each other. NMI measures how similar two partitions are and VI
measures how different two partitions are. Just pick one.

Last updated: Feb. 16 2023
Author(s): Xining Chen
"""
from collections import Counter
import networkx as nx
from math import log
from matplotlib import pyplot as plt
from tqdm import tqdm
import bct
import matplotlib.cm as cm
from sklearn.metrics.cluster import normalized_mutual_info_score


def matrix_to_network_format(W, output_path):
    """
    Converts an adjacency matrix to network format file: node1 node2 w12
    Some community detection algorithms only accept connectome input as a network format instead of adjacency matrix.
    Since a functional connectome is undirected and weighted, the network format will have the same edge twice with
    the same weight.
    :param W: adjacency matrix (2D numpy array or 2D list) to be converted
    :param output_path: path for the network format file.
    :return: n/a
    """
    N = W.shape[0]
    o = ""
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if W[i][j] == 0:
                continue
            o += f"{i} {j} {W[i][j]}\n"
    with open(output_path, "w") as f:
        f.write(o)


def get_clusters(partition):
    """
    Convert partition format to a list of clusters format
    :param partition: community structure breakdown (list with a community number at the node's index)
    :return: community cluster format, where the nodes within the same community are listed together.
    """
    c = Counter(partition)
    N = len(c)
    reindex = dict(zip(c.keys(), range(N)))
    clustering = [[] for i in range(N)]
    for node, k in enumerate(partition):
        clustering[reindex[k]].append(node)
    return clustering


def variation_of_information(X, Y):
    """
    Variation of information (VI).
    Meila, M. (2007). Comparing clusterings-an information
    based distance. Journal of Multivariate Analysis, 98,
    873-895. doi:10.1016/j.jmva.2006.11.013
    :param X:
    :param Y:
    :return:
    """
    n = float(len(X))
    cluster_1 = get_clusters(X)
    cluster_2 = get_clusters(Y)

    H_1 = 0.0
    H_2 = 0.0
    I = 0.0
    for i, c1 in enumerate(cluster_1):
        p_1 = len(c1) / n
        H_1 += p_1 * log(p_1, 2)
        for c2 in cluster_2:
            p_2 = len(c2) / n
            if i == 0:
                H_2 += p_2 * log(p_2, 2)
            r = float(len(set(c1) & set(c2))) / n
            if r > 0.0:
                I += r * log(r / (p_1 * p_2), 2)
    vi = -1.0 * H_1 + -1.0 * H_2 - 2.0 * I
    return vi


def community_stats(partitions_by_gamma, similarity_by_gamma):
    """

    :param partitions_by_gamma:
    :param similarity_by_gamma:
    :return:
    """
    rep_part = []
    avg_stability = []
    std_dev_stab = []
    variance_stab = []
    for r, sim_list in tqdm(similarity_by_gamma.items()):
        list_of_partitions = partitions_by_gamma[r]
        rep_part.append(get_best_partition(list_of_partitions))
        # avg_stability.append(stat.mean(sim_list))
        # std_dev_stab.append(stat.stdev(sim_list))
        # variance_stab.append(stat.variance(sim_list))
    return rep_part, avg_stability, std_dev_stab, variance_stab


def get_best_partition(partitions):
    """
    Get the best partition from a set of partitions. "Best" partition is defined here as the most similar partition to the ensemble for a particular gamma.
    :param partitions: list of partitions
    :return: the best (most similar by Normalized Mutual Information) partition
    """
    if len(partitions) == 1:
        return partitions[0]
    avg_similarity_per_partition = [0] * len(partitions)
    # Compute average similarity of 1 partition with everyone else.
    for i, current_partition in enumerate(partitions):
        for j in range(i + 1, len(partitions)):
            new_partition = partitions[j]
            nmi = normalized_mutual_info_score(current_partition, new_partition)
            avg_similarity_per_partition[i] += nmi
            avg_similarity_per_partition[j] += nmi
    avg_similarity_per_partition = [a / (len(partitions) - 1) for a in avg_similarity_per_partition]
    # Select the partition with the highest similarity
    max_sim = max(avg_similarity_per_partition)
    max_sim_i = avg_similarity_per_partition.index(max_sim)
    return partitions[max_sim_i]


def get_partitions(W, gamma, B='modularity', rep=2):
    """
    Run Louvain community detection algorithm. The Louvain community detection used here can be applied to networks with
    negative weights.
    :param W: Directed/undirected weighted/binary adjacency matrix.
    :param gamma: resolution parameter
    :param B: Various adaptation of Louvain. If you have negative weights, used 'negative_asym'.
    :param rep: How many times to repeat community detection for a specific gamma.
    :return: list of partitions, list of modularity for each partition
    """
    partitions = []
    modularity = []
    for i in range(rep):
        partition, q = bct.community_louvain(W, gamma=gamma, B=B)
        partitions.append(partition)
        modularity.append(q)
    return partitions, modularity


def draw_community(G, partition):
    """
    Draw the network G given it's community structure (partition).
    Only use this to get a sense if the community_by_modularity_stability function above is working.
    Avoid visualizing a brain network like this.
    :param G: network x graph object
    :param partition: community structure of G
    :return:
    """
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()

