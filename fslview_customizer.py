import matplotlib.pyplot as plt
import numpy as np
from colour import Color


color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 128, 0], 4: [0, 153, 77],
             5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [255, 255, 255]}
cortical_lut_f = 'Schaefer2018_1000Parcels_7Networks_order.txt'
subcortical_lut_f = 'Subcortical.nii.txt'
output_folder = '.'


def create_cortical_surfer_lut(partition, fname):
    """
    Creates a cortical lut that displays the community structure for free surfer/fslview.
    :param partition: community/module assignment list (list)
    :param fname: output file name (string)
    :return: n/a
    """
    f = open(cortical_lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split("\t")
        vec[2] = str(color_rgb[partition[l]][0])
        vec[3] = str(color_rgb[partition[l]][1])
        vec[4] = str(color_rgb[partition[l]][2])
        my_file_content += '\t'.join(vec)
    with open(f"{output_folder}/{fname}.txt", 'w') as output_file:
        output_file.write(my_file_content)


def create_subcortical_surfer_lut(partition, fname):
    """
    Creates a subcortical lut that displays the community structure for free surfer/fslview.
    :param partition: community/module assignment list (list)
    :param fname: output file name (string)
    :return: n/a
    """
    f = open(subcortical_lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")[:-1]
        my_file_content += f"{vec[0]} {round(color_rgb[partition[1000 + l]][0] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][1] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][2] / 255, 5)} " \
                           f"{vec[1]}\n"

    with open(f"{output_folder}/{fname} subcortex.txt", 'w') as output_file:
        output_file.write(my_file_content)


def export_cortical_surfer_lut_gradient(fn, nodes_freq, fname, plot_colors=False):
    """
    Creates a cortical LUT file that displays a single color gradient (to white) for visualizing in FreeSurfer or fslView.
    :param fn: functional network index - this is the single color based on the color_rgb dictionary. The gradient will
    be created from this color to white, with the highest frequency colored by the selected color and white for frequency
    of 0. (int, or whatever the key for the color_rgb)
    :param nodes_freq: frequency information to display as gradient (list or 1d numpy array of length of total number
    of nodes in the network with the value of each element being a counter/frequency integer)
    :param fname: output file name (string)
    :param plot_colors: if True then you will see a legend of the color gradient plotted, otherwise False
    :return: n/a
    """
    max_color = Color(rgb=tuple(np.array(color_rgb[fn])/255))
    gradient_colors = list(Color("white").range_to(max_color, int(max(nodes_freq))+1))
    f = open(cortical_lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split("\t")
        color = np.array(gradient_colors[int(nodes_freq[l])].rgb)*255
        vec[2] = str(int(color[0]))
        vec[3] = str(int(color[1]))
        vec[4] = str(int(color[2]))
        my_file_content += '\t'.join(vec)
    with open(f"{output_folder}/{fname}.txt", 'w') as output_file:
        output_file.write(my_file_content)

    if plot_colors:
        # Display a plot with a legend of the colors
        for ci, mc in enumerate(gradient_colors):
            plt.scatter([1], [1], color=mc.web, label=f"{ci} cycles")
        plt.legend(prop = { "size": 20 }, markerscale=3)
        plt.show()


def export_subcortical_surfer_lut_gradient(fn, nodes_freq, fname):
    """
    Creates a subcortical LUT file that displays a single color gradient (to white) for visualizing in FreeSurfer or fslView.
    :param fn: functional network index - this is the single color based on the color_rgb dictionary. The gradient will
    be created from this color to white, with the highest frequency colored by the selected color and white for frequency
    of 0. (int, or whatever the key for the color_rgb)
    :param nodes_freq: frequency information to display as gradient (list or 1d numpy array of length of total number
    of nodes in the network with the value of each element being a counter/frequency integer)
    :param fname: output file name (string)
    :return: n/a
    """
    max_color = Color(rgb=tuple(np.array(color_rgb[fn])/255))
    gradient_colors = list(Color("white").range_to(max_color, int(max(nodes_freq))+1))
    f = open(subcortical_lut_f, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        color = gradient_colors[int(nodes_freq[l])].rgb
        vec = line.split(" ")[:-1]
        my_file_content += f"{vec[0]} {color[0]} " \
                           f"{color[1]} " \
                           f"{color[2]} " \
                           f"{vec[1]}\n"
    with open(f"{output_folder}/{fname} subcortex.txt", 'w') as output_file:
        output_file.write(my_file_content)