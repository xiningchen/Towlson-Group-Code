from collections import Counter
import networkx as nx
from math import log
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import bct
import matplotlib.cm as cm
from sklearn.metrics.cluster import normalized_mutual_info_score
import math

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
        p_1 = len(c1)/n
        H_1 += p_1 * log(p_1, 2)
        for c2 in cluster_2:
            p_2 = len(c2)/n
            if i == 0:
                H_2 += p_2 * log(p_2, 2)
            r = float(len(set(c1) & set(c2))) / n
            if r > 0.0:
                I += r * log(r / (p_1 * p_2), 2)
    vi = -1.0*H_1 + -1.0*H_2 - 2.0*I
    return vi

# Returns the optimal gamma to use. Factors considered:
# (1) The number of communities detected plateaus
# (2) The avg. similarity value plateaus
# Avg. similarity value = the average similarity value between comparing gamm_i with gamma_{i-1} and gamma_{i+1}
# Select the start of gamma where the largest plateau begins
# def best_gamma(list_of_partitions):
#     avg_similarities = []
#     comm_count = []
#     for i in range(len(list_of_partitions)):
#         current_partition = list_of_partitions[i]
#         comm_count.append(len(Counter(current_partition)))
#         if i == 0:
#             next_partition = list_of_partitions[i + 1]
#             avg_sim = normalized_mutual_info_score(current_partition, next_partition)
#         elif i == len(list_of_partitions)-1:
#             prev_partition = list_of_partitions[i - 1]
#             avg_sim = normalized_mutual_info_score(current_partition, prev_partition)
#         else:
#             next_partition = list_of_partitions[i + 1]
#             prev_partition = list_of_partitions[i - 1]
#             avg_sim = 0.5*(normalized_mutual_info_score(current_partition, next_partition)
#                            + normalized_mutual_info_score(current_partition, prev_partition))
#         avg_similarities.append(avg_sim)
#     return avg_similarities, comm_count

# return the best partition from a set of partitions.
# "Best" partition is defined here as the most similar partition to the ensemble for a particular gamma.
def get_best_partition(partitions):
    avg_similarity_per_partition = [0]*len(partitions)
    # Compute average similarity of 1 partition with everyone else.
    for i, current_partition in enumerate(partitions):
        for j in range(i+1, len(partitions)):
            new_partition = partitions[j]
            nmi = normalized_mutual_info_score(current_partition, new_partition)
            avg_similarity_per_partition[i] += nmi
            avg_similarity_per_partition[j] += nmi
    avg_similarity_per_partition = [a/(len(partitions)-1) for a in avg_similarity_per_partition]
    # Select the partition with the highest similarity
    max_sim = max(avg_similarity_per_partition)
    max_sim_i = avg_similarity_per_partition.index(max_sim)
    return partitions[max_sim_i]

def community_by_modularity_stability(W, res_lower_range, res_upper_range, inc, B='modularity', rep=2):
    # count = int((rep*(rep-1))/2)
    r_upper = round((res_upper_range - res_lower_range)/inc)
    partition_by_gamma = {r: [] for r in range(0, r_upper)}
    similarity_by_gamma = {r: [] for r in range(0, r_upper)}
    modularity_by_gamma_per_partition = {r: [] for r in range(0, r_upper)}
    for r in tqdm(range(0, r_upper)):
        for i in range(0, rep):
            res = res_lower_range + inc * r
            partition, q = bct.community_louvain(W, gamma=res, B=B)
            if len(partition_by_gamma[r]) >= 1:
                # Compute distance between previous partition and current partition
                pre_partition = partition_by_gamma[r][i-1]
                similarity_by_gamma[r].append(normalized_mutual_info_score(pre_partition, partition))
                # for plist in partition_by_gamma[r]:
                #     similarity_by_gamma[r].append(normalized_mutual_info_score(plist, partition))
            partition_by_gamma[r].append(partition)
            modularity_by_gamma_per_partition[r].append(q)
    return partition_by_gamma, similarity_by_gamma, modularity_by_gamma_per_partition

def draw_community(G, partition):
    # print(community_louvain.modularity(partition, G))
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
