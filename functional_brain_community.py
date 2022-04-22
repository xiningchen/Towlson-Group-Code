# !pip install python-louvain
from collections import Counter

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from community import community_louvain
import matplotlib.cm as cm
from sklearn.metrics.cluster import adjusted_rand_score


def find_resolution_range(G, min_community_num, max_community_num):
    x_axis = []
    y_axis = []
    good_res = []
    for r in tqdm(range(0, 300)):
        res = 0.1+r*0.01
        x_axis.append(res)
        c = community_louvain.best_partition(G, resolution=res, randomize=True)
        if len(Counter(c.values())) >= min_community_num and len(Counter(c.values())) <= max_community_num:
            avg_num = len(Counter(c.values()))
            for i in range(0,100):
                c = community_louvain.best_partition(G, resolution=res, randomize=True)
                avg_num += len(Counter(c.values()))
            avg_num = avg_num/100
            y_axis.append(avg_num)
            if avg_num >= min_community_num and avg_num <= max_community_num:
                good_res.append((res, avg_num))
        else:
            y_axis.append(len(Counter(c.values())))
    return good_res, x_axis, y_axis


def community_by_modularity_stability(G, count, res_upper_range, res_lower_range):
    partition_by_gamma = {r: [] for r in range(0, res_upper_range)}
    partition_list_by_gamma = {r: [] for r in range(0, res_upper_range)}
    stability_by_gamma = {r: 0.0 for r in range(0, res_upper_range)}
    modularity_by_gamma = {r: 0.0 for r in range(0, res_upper_range)}
    for i in tqdm(range(0, 500)):
        for r in range(0, res_upper_range):
            res = res_lower_range + (0.005) * r
            partition = community_louvain.best_partition(G, resolution=res, randomize=True)
            partition_list = [p for nlab, p in partition.items()]
            if len(partition_by_gamma[r]) >= 1:
                # Compute a z-rand score
                for plist in partition_list_by_gamma[r]:
                    stability_by_gamma[r] += adjusted_rand_score(plist, partition_list)
                    if r == 0:
                        count += 1
            partition_by_gamma[r].append(partition)
            partition_list_by_gamma[r].append(partition_list)
            # Compute modularity value
            modularity_by_gamma[r] += community_louvain.modularity(partition, G)
    # Calculate average z-rand score per gamma
    avg_stab_by_gamma = np.array([z / count for r, z in stability_by_gamma.items()])
    # Calculate average modularity score per gamma
    avg_modularity_by_gamma = np.array([mod / 500 for r, mod in modularity_by_gamma.items()])
    # Calculate weighted avg z-rand = <mod>*<z-rand> per gamma
    weighted_avg_zscore = avg_modularity_by_gamma * avg_stab_by_gamma
    # Find max(weighted avg z-rand) --> gamma_max
    index_max = np.argmax(weighted_avg_zscore)
    if weighted_avg_zscore[index_max] != max(weighted_avg_zscore):
        print("Max gamma incorrect - something wrong.")
    return res_lower_range+0.005*index_max, index_max, avg_modularity_by_gamma, avg_stab_by_gamma, partition_by_gamma


def draw_community(G, partition):
    print(community_louvain.modularity(partition, G))
    pos = nx.spring_layout(G)
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()