"""
Functions for creating network x graphs using brain connectome data and computing a few network metrics given a
network.

Last updated: Feb. 16, 2023
Author(s): Xining Chen
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import statistics as stats


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


def get_community_to_node_map(communities, nodeMetaData):
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

