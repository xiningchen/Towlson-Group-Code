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
# ----------------------------------------------------------------------------------------------
# Inputs:
# ----------------------------------------------------------------------------------------------
def get_network_stats(G):
