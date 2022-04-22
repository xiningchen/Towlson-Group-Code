###############################################################################################
# Supportive functions for file input and output from .xlsx, .mat, .csv
# See data_io_example.ipynb for example code using these functions.
# Last updated: April 17, 2022
# Author(s): Xining Chen
###############################################################################################
import os
import pickle as pkl
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.io import loadmat

# Some global variables
BRAINNET_PATH = 'BrainNet_Viewer/'
DATE = datetime.today().strftime('%Y-%m-%d')

# --------------------------------------------------------------------------------------------- File IO
# Read data from an .xlsx file
# ----------------------------------------------------------------------------------------------
# Inputs:
# path (string): a file path to the folder that contains the file(s) to be read
# file_name (string): optional input with default value None. If the value is None, then function will
#                     load every .xlsx file in the path specified. Otherwise, it will open the file_name
#                     specified from user input.
# numpyArray (boolean): optional flag for deciding if the read input should be returned as a Panda
#                       dataframe or a numpy array. Default value is False (return as Panda Dataframes)
# ----------------------------------------------------------------------------------------------
def import_XLSX(path, file_name=None, numpy_array=False):
    if file_name is None:
        files = {}
        for root, dirs, files in os.walk(path):
            for file in tqdm(files):
                if not file.endswith('.xlsx'):
                    continue
                if numpy_array is False:
                    files[file] = pd.read_excel(path + file, index_col=0, header=0)
                if numpy_array is True:
                    mat = pd.read_excel(path + file, index_col=0, header=0)
                    files[file] = mat.to_numpy()
        return files
    else:
        if numpy_array is False:
            return pd.read_excel(path + file_name, index_col=0, header=0)
        if numpy_array is True:
            mat = pd.read_excel(path + file_name, index_col=0, header=0)
            return mat.to_numpy()

# Read data from a .mat file
# ----------------------------------------------------------------------------------------------
# Inputs:
# path (string): a file path to the folder that contains the file(s)
# file_name (string): optional input with default value None. If the value is None, then function will
#                     load every .xlsx file in the path specified. Otherwise, it will open the file_name
#                     specified from user input.
# numpyArray (boolean): optional flag for deciding if the read input should be returned as a Panda
#                       dataframe or a numpy array. Default value is False (return as Panda Dataframes)
# ----------------------------------------------------------------------------------------------
def import_MAT(path, file_name=None):
    if file_name is None:
        MAT_files = []
        for root, dirs, files in os.walk(path):
            for file in tqdm(files):
                if not file.endswith('.mat'):
                    continue
                data = loadmat(root+file)
                MAT_files.append(data)
        return MAT_files
    else:
        data = loadmat(path+file_name)
        print(data.keys())
        return data


# --------------------------------------------------------------------------------------------- Pickling Data
# Pickles data stored in the data variable
# ----------------------------------------------------------------------------------------------
# Inputs:
#
# ----------------------------------------------------------------------------------------------
def save_to_pickle(data, path, pickle_name):
    pickle_name += f'-{DATE}.pkl'
    file_path = os.path.join(path, pickle_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, 'wb') as f:
        pkl.dump(data, f)

    f.close()
    print(f"Saved to {file_path}.")

def load_from_pickle(path, pickle_name):
    file_path = os.path.join(path, pickle_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            res = pkl.load(f)
        return res
    else:
        return []

# --------------------------------------------------------------------------------------------- BrainNet Viewer
# Function for exporting a .node file for BrainNet Viewer
# .node file defined as an ASCII text file with suffix 'node'
# There are 6 columns: 1-3 represent node coordinates,
# col. 4 represents node color, col. 5 represents node size, last col. represent node label
# ----------------------------------------------------------------------------------------------
# Inputs:
# nodeAttrDict: a dictionary of node labels with corresponding values for node color and node size
# nodeDF: node data frame containing the node's X, Y, Z coordinates.
# fileName: exported file .node name.
# ----------------------------------------------------------------------------------------------
def export_node_file(node_df, nodeAttrDict, file_name='tempNodeFileName'):
    file_name = BRAINNET_PATH + file_name + ".node"
    with open(file_name, 'w') as writer:
        lines = []
        for nodeLabel, row in node_df.iterrows():
            color = nodeAttrDict['color'][nodeLabel]
            size = nodeAttrDict['size'][nodeLabel]
            lines.append(f"{row['X']}\t{row['Y']}\t{row['Z']}\t{color}\t{size}\t{nodeLabel}\n")
        writer.writelines(lines)
    print(f"File saved to {file_name}")


# ----------------------------------------------------------------------------------------------
# Function for exporting a .edge file for BrainNet Viewer
# ----------------------------------------------------------------------------------------------
# Inputs:
# MIGHT NOT NEED ----
# ----------------------------------------------------------------------------------------------
def export_edge_file(adj_df, file_name='tempNodeFileName', binarize=True):
    file_name = BRAINNET_PATH + file_name + ".edge"
    with open(file_name, 'w') as writer:
        lines = []
        for nodeLabel, row in adj_df.iterrows():
            line = ""
            for c in row:
                if binarize:
                    if c>0:
                        line += "1\t"
                    else:
                        line += "0\t"
                else:
                    line += f"{c}\t"
            line.rstrip()
            line += "\n"
            lines.append(line)
        writer.writelines(lines)
    print(f"File saved to {file_name}")
