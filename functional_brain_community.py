from collections import Counter
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
# from community import community_louvain
import bct
import matplotlib.cm as cm
from sklearn.metrics.cluster import adjusted_rand_score
import math

def find_resolution_range(W, min_community_num, max_community_num, min_gamma, max_gamma, inc, B='modularity', sample_size=1):
    x_axis = []
    y_axis = []
    good_res = []
    max_range = math.ceil((max_gamma - min_gamma) / inc)
    for r in tqdm(range(0, max_range)):
        res = min_gamma + r * inc
        x_axis.append(res)
        # c = community_louvain.best_partition(G, resolution=res, randomize=True)
        c, q = bct.community_louvain(W, gamma=res, B=B)
        avg_num = len(Counter(c))
        if avg_num >= min_community_num and avg_num <= max_community_num:
            for i in range(0,sample_size):
                c, q = bct.community_louvain(W, gamma=res, B=B)
                avg_num += len(Counter(c))
            avg_num = avg_num/sample_size
            y_axis.append(avg_num)
            if avg_num >= min_community_num and avg_num <= max_community_num:
                good_res.append((res, avg_num))
        else:
            y_axis.append(len(Counter(c)))
    return good_res, x_axis, y_axis


def community_by_modularity_stability(W, res_lower_range, res_upper_range, inc, B='modularity', rep=2):
    count = math.comb(rep, 2)
    r_upper = math.ceil((res_upper_range - res_lower_range)/inc)
    partition_by_gamma = {r: [] for r in range(0, r_upper)}
    partition_list_by_gamma = {r: [] for r in range(0, r_upper)}
    stability_by_gamma = {r: 0.0 for r in range(0, r_upper)}
    modularity_by_gamma = {r: 0.0 for r in range(0, r_upper)}
    for i in tqdm(range(0, rep)):
        for r in range(0, r_upper):
            res = res_lower_range + inc * r
            # partition = community_louvain.best_partition(G, resolution=res, randomize=True)
            partition, q = bct.community_louvain(W, gamma=res, B=B)
            partition_list = [p for p in partition]
            if len(partition_by_gamma[r]) >= 1:
                # Compute a z-rand score
                for plist in partition_list_by_gamma[r]:
                    stability_by_gamma[r] += adjusted_rand_score(plist, partition_list)
            partition_by_gamma[r].append(partition)
            partition_list_by_gamma[r].append(partition_list)
            # Compute modularity value  ****** is q returned modularity value? Or something q-statistc??
            modularity_by_gamma[r] += q
    # Calculate average z-rand score per gamma
    avg_stab_by_gamma = np.array([z / count for r, z in stability_by_gamma.items()])
    # Calculate average modularity score per gamma
    avg_modularity_by_gamma = np.array([mod / rep for r, mod in modularity_by_gamma.items()])
    # Calculate weighted avg z-rand = <mod>*<z-rand> per gamma
    weighted_avg_zscore = avg_modularity_by_gamma * avg_stab_by_gamma
    # Find max(weighted avg z-rand) --> gamma_max
    index_max = np.argmax(weighted_avg_zscore)
    if weighted_avg_zscore[index_max] != max(weighted_avg_zscore):
        print("Max gamma incorrect - something wrong.")
    return res_lower_range + inc*index_max, index_max, avg_modularity_by_gamma, avg_stab_by_gamma, partition_by_gamma


def draw_community(G, partition):
    # print(community_louvain.modularity(partition, G))
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()