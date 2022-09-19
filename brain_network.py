###############################################################################################
# Random functions for dealing with brain connectomes.
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
from scipy.io import loadmat
import networkx as nx

# Global variables
DATE = datetime.today().strftime('%Y-%m-%d')

# Get an averaged connectome from a set of connectome data.
# ----------------------------------------------------------------------------------------------
# Inputs:
# path (string): a file path to the folder that contains the connectomes to be averaged
# file_name (string): optional input with default value None. If the value is None, then function will
#                     load every .xlsx file in the path specified. Otherwise, it will open the file_name
#                     specified from user input.
# numpyArray (boolean): optional flag for deciding if the read input should be returned as a Panda
#                       dataframe or a numpy array. Default value is False (return as Panda Dataframes)
# ----------------------------------------------------------------------------------------------
# Note: There's no filtering or sanity check of whether a connectome is "valid" or not.
# Modify the read_excel() function to read additional rows and columns. Currently set to 94x94 matrix.
def get_avg_connectome(dir_path, shape):
    total_connectome = np.zeros(shape=shape)
    num_files = 0
    # Load every .mat file in MATLAB_FILE_BASE and average them
    for root, dirs, files in os.walk(dir_path):
        for file in tqdm(files):
            if not file.endswith('.xlsx'):
                continue
            brain = pd.read_excel(root + file, index_col = 0, header = 0, nrows=94, usecols="A:CQ")
            brain_numpy_array = brain.to_numpy()
            num_files += 1
            total_connectome = total_connectome + brain_numpy_array
    return total_connectome/num_files


# Given a network G, print basic network statistics.
# Network stats:
# (1) Average weight (if weighted) (float)
# (2) Average degree (float)
# (3) Degree distribution
# (4) Weight distribution (if weighted) (list)
# (5) Average path length (float)
# (6) Average clustering coefficient (float)
# ----------------------------------------------------------------------------------------------
# Input(s): NetworkX graph G (can be weighted or unweighted)
# Output(s): Dictionary
# ----------------------------------------------------------------------------------------------
def get_network_stats(G):
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