"""
Supportive functions for file input and output from .xlsx, .mat, .csv
Last updated: Feb. 16 2023
Author(s): Xining Chen
"""
import os
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat


def __check_path(full_path):
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))


def import_XLSX(path, file_name=None, numpy_array=False, index_col=0, sheet_name="Sheet1"):
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
                    files[file] = pd.read_excel(path + file, index_col=index_col, header=0, sheet_name=sheet_name)
                if numpy_array is True:
                    mat = pd.read_excel(path + file, index_col=index_col, header=0, sheet_name=sheet_name)
                    files[file] = mat.to_numpy()
        return files
    else:
        if numpy_array is False:
            return pd.read_excel(path + file_name, index_col=index_col, header=0, sheet_name=sheet_name)
        if numpy_array is True:
            mat = pd.read_excel(path + file_name, index_col=index_col, header=0, sheet_name=sheet_name)
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
                data = loadmat(root + file)
                MAT_files.append(data)
                MAT_files_name.append(file)
        return MAT_files_name, MAT_files
    else:
        data = loadmat(path + file_name)
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
    """
    Look for some file.
    :param path: directory to search in.
    :param keyword: word or file name to look for in the file name.
    :param ext: the file extension of the file you're looking for
    :return:
    """
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
    """
    Load all pickles at some path.
    :param path: path to a folder.
    :param keyword: keyword to check if file's name contains this word.
    :return:
    """
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
