###############################################################################################
# Supportive functions for file input and output from .xlsx, .mat, .csv
# See data_io_example.ipynb for example code using these functions.
# Last updated: April 17, 2022
# Author(s): Xining Chen
###############################################################################################
import os
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.io import loadmat

def __check_path(full_path):
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))

def import_XLSX(path, file_name=None, numpy_array=False, index_col=0):
    """
    Read data from an .xlsx file
    :param path: file path to the folder that contains the file(s) to be read
    :param file_name: optional input with default value None. If the value is None, then function will
    load every .xlsx file in the path specified. Otherwise, it will open the file_name specified from user input.
    :param numpy_array: optional flag for deciding if the read input should be returned as a Panda dataframe or
    a numpy array. Default value is False (return as Panda Dataframes)
    :param index_col: Specify if there's an index column or not when reading in.
    :return: Read in data from excel.
    """
    if file_name is None:
        files = {}
        for root, dirs, files in os.walk(path):
            for file in tqdm(files):
                if not file.endswith('.xlsx'):
                    continue
                if numpy_array is False:
                    files[file] = pd.read_excel(path + file, index_col=index_col, header=0)
                if numpy_array is True:
                    mat = pd.read_excel(path + file, index_col=index_col, header=0)
                    files[file] = mat.to_numpy()
        return files
    else:
        if numpy_array is False:
            return pd.read_excel(path + file_name, index_col=index_col, header=0)
        if numpy_array is True:
            mat = pd.read_excel(path + file_name, index_col=index_col, header=0)
            return mat.to_numpy()

def import_MAT(path, file_name=None):
    """
    Read data from a .mat file
    :param path: file path to the folder that contains the file(s)
    :param file_name: optional input with default value None. If the value is None, then function will load every .xlsx
    file in the path specified. Otherwise, it will open the file_name specified from user input.
    :return: data read from .MAT files
    """
    if file_name is None:
        MAT_files_name = []
        MAT_files = []
        for root, dirs, files in os.walk(path):
            for file in tqdm(files):
                if not file.endswith('.mat'):
                    continue
                data = loadmat(root+file)
                MAT_files.append(data)
                MAT_files_name.append(file)
        return MAT_files_name, MAT_files
    else:
        data = loadmat(path+file_name)
        print(data.keys())
        return data

def save_to_pickle(data, path, pickle_name):
    """
    Save some data to a pickle.
    :param data: data to be saved
    :param path: path to save location
    :param pickle_name: name of pickle file
    :return:
    """
    if '.pkl' not in pickle_name:
        pickle_name += '.pkl'
    file_path = os.path.join(path, pickle_name)
    __check_path(file_path)
    with open(file_path, 'wb') as f:
        pkl.dump(data, f)
    f.close()
    print(f"Saved to {file_path}.")

def load_from_pickle(path, pickle_name):
    """
    Load data from a pickle
    :param path: path to file location
    :param pickle_name: pickle name to be read
    :return: data stored in pickle
    """
    file_path = os.path.join(path, pickle_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            res = pkl.load(f)
        return res
    else:
        return []

def find_file(path, keyword, ext='.pkl'):
    for root, dirs, files in os.walk(path):
        for file in (files):
            if not file.endswith(ext):
                continue
            if keyword not in file:
                continue

            with open(root + file, 'rb') as f:
                data = pkl.load(f)
            return data, file
    print("Could not find file.")
    return 0

def load_all_pickles(path, keyword=""):
    all_data = {}
    for root, dirs, files in os.walk(path):
        for file in (files):
            if not file.endswith('.pkl'):
                continue
            if keyword == "":
                with open(root + file, 'rb') as f:
                    all_data[file] = pkl.load(f)
            elif keyword in file:
                with open(root + file, 'rb') as f:
                    all_data[file] = pkl.load(f)
    return all_data
# --------------------------------------------------------------------------------------------- BrainNet Viewer
# Function for exporting a .node file for BrainNet Viewer
# .node file defined as an ASCII text file with suffix 'node'
# There are 6 columns: 1-3 represent node coordinates,
# col. 4 represents node color, col. 5 represents node size, last col. represent node label
# ----------------------------------------------------------------------------------------------
# Inputs:
# nodeAttrDict: a dictionary of node labels with corresponding values for node color and node size
# node_df: node dataframe containing the node's X, Y, Z coordinates. Dataframe's index are node names.
# fileName: exported file .node name.
# ----------------------------------------------------------------------------------------------
def export_node_file(node_df, color, size, path, file_name='tempNodeFileName'):
    """
    Function for exporting a .node file for BrainNet Viewer.
    .node file defined as an ASCII text file with suffix 'node'
    There are 6 columns: 1-3 represent node coordinates,
    col. 4 represents node color, col. 5 represents node size, last col. represent node label
    :param node_df:
    :param color:
    :param size:
    :param path:
    :param file_name:
    :return:
    """
    file_name += '.node'
    file_path = os.path.join(path, file_name)
    __check_path(file_path)
    with open(file_path, 'w') as writer:
        lines = []
        for node_name, row in node_df.iterrows():
            lines.append(f"{row['X']}\t{row['Y']}\t{row['Z']}\t{color[node_name]}\t{size[node_name]}\t{node_name}\n")
        writer.writelines(lines)
    print(f"File saved to {file_name}")


def export_edge_file(adj_df, path, file_name='tempNodeFileName', binarize=True):
    file_name += '.edge'
    file_path = os.path.join(path, file_name)
    __check_path(file_path)
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
