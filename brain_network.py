###############################################################################################
# Functions for dealing with brain connectomes.
#
# Last updated: Mar. 10, 2022
# Author(s): Xining Chen
###############################################################################################
import os
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import networkx as nx
import fsleyes_customizer as fsleyes
import statistics as stats
from collections import Counter
import data_io as myFunc

# Global variables
DATE = datetime.today().strftime('%Y-%m-%d')

def get_avg_connectome(dir_path, shape):
    """
    Get an averaged connectome from a set of connectome data stored in .xlsx type files
    Modify the read_excel() function to read additional rows and columns. Currently set to 94x94 matrix.
    :param dir_path: file path to the folder that contains the connectomes to be averaged
    :param shape: connectome shape
    :return: a single average connectome
    """
    total_connectome = np.zeros(shape=shape)
    num_files = 0
    for root, dirs, files in os.walk(dir_path):
        for file in tqdm(files):
            if not file.endswith('.xlsx'):
                continue
            brain = pd.read_excel(root + file, index_col = 0, header = 0, nrows=94, usecols="A:CQ")
            brain_numpy_array = brain.to_numpy()
            num_files += 1
            total_connectome = total_connectome + brain_numpy_array
    return total_connectome/num_files

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
    result['avg_degree'] = sum(degree_seq)/len(degree_seq)
    result['degree_seq'] = degree_seq
    # Weighted distribution
    attributes = nx.get_edge_attributes(G, 'weight')
    if len(attributes) != 0:
        weight_seq = sorted([G.degree(node, weight = "weight") for node in G], reverse=True)
        result['avg_weight'] = sum(weight_seq)/len(weight_seq)
        result['weight_seq'] = weight_seq
    return result

def get_network_stats2(G):
    result = {'avg_path_length': nx.average_shortest_path_length(G),
              'clustering': np.mean(list(nx.clustering(G, weight="weight").values())),
              'shortest_path': nx.average_shortest_path_length(G),
              'global_efficiency': nx.global_efficiency(G)}
    node_strengths = sorted([(node, G.degree(node, weight="weight")) for node in G.nodes()], reverse=True, key=lambda x: x[1])
    avg_node_strength = [x[1] for x in node_strengths]
    result['avg_weight'] = sum(avg_node_strength) / len(avg_node_strength)
    return result

def get_community_to_node_map(communities, nodeMetaData):
    nodeList = list(nodeMetaData.index)
    community_to_node_list = {c: [] for c in communities.values()}
    for n in nodeList:
        community_to_node_list[communities[nodeMetaData['Functional_System'][n]]].append(n)
    return community_to_node_list

# Calculate WEIGHTED z-score of each node for a graph G. Returns zscore data in groups.
def get_zscore(G, communities, community_to_node_list):
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

# Calculate WEIGHTED PC of each node in a graph G.
def get_PC(G, nodeList, community_to_node_list):
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
            tot += (w_is/w_i)**2
        if (tot > 1):
            print(f"ERROR - {node}")
            break
        pc[node] = 1 - tot
        pc_attrs[node]["pc"] = 1 - tot
    nx.set_node_attributes(G, pc_attrs)
    return pc, G


def export_for_fsleyes(project, partition, fname, btype, reindex="top8"):
    DATA_PATH = f'../{project}/data/'
    FILE = fname + '.txt'
    # ---- PART 1
    if reindex == "top8":
        x = [p[0] for p in Counter(partition).most_common()]
        # RE-INDEX Community # from 1 = largest community, to smaller communities
        reindex_map = dict(zip(x, np.arange(1, len(Counter(partition)) + 1)))
        reindex_partition = [reindex_map[c] for c in partition]
    if reindex == "+1":
        reindex_partition = [c+1 for c in partition]
    else:
        reindex_partition = partition
    # ---- PART 2
    node_cog_df = myFunc.import_XLSX(DATA_PATH, 'node_cog.xlsx')
    node_list = list(node_cog_df['region_name'])
    node_list_2 = [n.replace("_", "-") for n in node_list]

    data_formatted = dict(zip(node_list_2, reindex_partition))
    # Export to txt in format described above
    buffer = ""
    for n, c in data_formatted.items():
        buffer += n + " " + str(c) + "\n"
    with open(f'../{project}/Brain_Atlas/data_to_plot/' + FILE, 'w') as f:
        f.write(buffer)

    # ---- PART 3
    if btype == 'both':
        types = ['cortical', 'subcortical']
    else:
        types = [btype]
    for btype in types:
        # Check file path
        os.path.abspath(f"../{project}/Brain_Atlas/data_to_plot/" + FILE)
        absolute_path = f"/Users/shine/Documents/MSc/Neuro Research/{project}/Brain_Atlas/"
        data_file_path = absolute_path + 'data_to_plot/'
        output_path = absolute_path + 'fsleyes_custom/'

        if btype == 'cortical':
            lut_file = absolute_path + 'Cortical.nii.txt'
            nii_file = absolute_path + 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
        if btype == 'subcortical':
            lut_file = absolute_path + 'Subcortical.nii.txt'
            nii_file = absolute_path + 'Tian_Subcortex_S4_3T_1mm.nii.gz'

        # In my case I need to call formatter.
        data_txt_file = absolute_path + 'data_to_plot/' + FILE
        formatted_data = fsleyes.format_data(data_txt_file, lut_file)
        fsleyes.run_customizer(output_path, lut_file, nii_file, fname=f'{FILE[:len(FILE) - 4]}_{btype}',
                               data_values=formatted_data)

def create_cortical_lut(partition, fname):
    color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 179, 102], 4: [0, 153, 77],
                 5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [0,0,0]}
    lut_f = '../Ovarian_hormone/Brain_Atlas/Schaefer2018_1000Parcels_7Networks_order.lut'
    f = open(lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")
        for i in range(1, 4):
            vec[i] = str(round(color_rgb[partition[l]][i - 1] / 255, 5))
        my_file_content += ' '.join(vec)
    with open(f"../Ovarian_hormone/Brain_Atlas/custom_lut/{fname}.lut", 'w') as output_file:
        output_file.write(my_file_content)

def create_subcortical_lut(partition, fname):
    color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 179, 102], 4: [0, 153, 77],
                 5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [0,0,0]}
    lut_f = '../Ovarian_hormone/Brain_Atlas/Subcortical.nii.txt'
    f = open(lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")[:-1]
        my_file_content += f"{vec[0]} {round(color_rgb[partition[1000 + l]][0] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][1] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][2] / 255, 5)} " \
                           f"{vec[1]}\n"

    with open(f"../Ovarian_hormone/Brain_Atlas/custom_lut/{fname}_subcortex.lut", 'w') as output_file:
        output_file.write(my_file_content)