from collections import Counter
import networkx as nx
from math import log
from matplotlib import pyplot as plt
from tqdm import tqdm
import bct
import matplotlib.cm as cm
from sklearn.metrics.cluster import normalized_mutual_info_score
import statistics as stat


# Given a list of partition, convert format to a list of clusters format
def get_clusters(partition):
    c = Counter(partition)
    N = len(c)
    reindex = dict(zip(c.keys(), range(N)))
    clustering = [[] for i in range(N)]
    for node, k in enumerate(partition):
        clustering[reindex[k]].append(node)
    return clustering


# Variation of information (VI)
#
# Meila, M. (2007). Comparing clusterings-an information
#   based distance. Journal of Multivariate Analysis, 98,
#   873-895. doi:10.1016/j.jmva.2006.11.013
def variation_of_information(X, Y):
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
    # For each gamma, plot the list of similarity values (NMI)
    rep_part = []
    avg_stability = []
    std_dev_stab = []
    variance_stab = []
    for r, sim_list in tqdm(similarity_by_gamma.items()):
        list_of_partitions = partitions_by_gamma[r]
        # avg_communities = 0
        # for part in list_of_partitions:
        #     avg_communities += len(Counter(part))
        # avg_communities = avg_communities / len(list_of_partitions)
        # if avg_communities < 6 or avg_communities > 25:
        #     continue
        rep_part.append(get_best_partition(list_of_partitions))
        # avg_stability.append(stat.mean(sim_list))
        # std_dev_stab.append(stat.stdev(sim_list))
        # variance_stab.append(stat.variance(sim_list))
    return rep_part, avg_stability, std_dev_stab, variance_stab


# return the best partition from a set of partitions.
# "Best" partition is defined here as the most similar partition to the ensemble for a particular gamma.
def get_best_partition(partitions):
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

def community_by_modularity_stability(W, g, B='modularity', rep=2):
    count = 0
    stability = 0
    modularity = []
    partitions = []
    for i in range(rep):
        partition, q = bct.community_louvain(W, gamma=g, B=B)
        partitions.append(partition)
        modularity.append(q)
        if len(partitions) > 1:
            for p in partitions:
                stability += normalized_mutual_info_score(p, partition)
                count += 1
    return partitions, modularity, stability

def draw_community(G, partition):
    # print(community_louvain.modularity(partition, G))
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
