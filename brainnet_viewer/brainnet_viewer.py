"""
Export functions for generating BrainNet Viewer input files. BrainNet Viewer accepts two types of files:
(1) .node: This file contains node information to be drawn. i.e. color, size, location (Center of Gravity, CoG)
(2) .edge: This file contains edge information to be drawn. It can be weighted (thickness of an edge), or just binary (exist or doesn't exist an edge)

Last updated: Feb. 16, 2023
Author(s): Xining Chen
"""
import os


def __check_path(full_path):
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))


def export_node_file(path, node_df, color=None, size=None, file_name='you_should_name_your_files'):
    """
    Function for exporting a .node file for BrainNet Viewer.
    .node file defined as an ASCII text file with suffix 'node'
    There are 6 columns: 1-3 represent node coordinates,
    col. 4 represents node color, col. 5 represents node size, last col. represent node label
    :param node_df: node dataframe containing the node's X, Y, Z coordinates. Dataframe's index are node names.
    :param color: dictionary with dataframe's index (node names) are the keys and corresponding value is the color of that node.
    :param size: dictionary with dataframe's index (node names) are the keys and corresponding value is the size of that node.
    :param path: output path for the .node file
    :param file_name: name of the .node file
    :return:
    """
    file_name += '.node'
    file_path = os.path.join(path, file_name)
    __check_path(file_path)
    if color is None:
        color = {x: 1 for x in node_df.index}
    if size is None:
        size = {x: 1 for x in node_df.index}

    with open(file_path, 'w') as writer:
        lines = []
        for node_name, row in node_df.iterrows():
            nn = node_name.replace("_", "-")
            lines.append(f"{row['X']}\t{row['Y']}\t{row['Z']}\t{color[node_name]}\t{size[node_name]}\t{nn}\n")
        writer.writelines(lines)
    print(f"File saved to {file_name}")


def export_edge_file(adj_df, path, file_name='you_should_name_your_files', binarize=True):
    """
    This should export a correct edge file for BrainNet Viewer. I think I tested the code and it works. But I didn't end up using it.
    :param adj_df: adjacency list dataframe (?)
    :param path:
    :param file_name:
    :param binarize:
    :return:
    """
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
