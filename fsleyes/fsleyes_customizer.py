import os
import SimpleITK as sitk
from collections import Counter
from .. import data_io as myFunc
import numpy as np


def change_image(image_path, data_values, rois):
    itk_img = sitk.ReadImage(image_path)
    img = sitk.GetArrayFromImage(itk_img)
    new_img = img.copy()
    new_img.fill(0)

    for roi, data_val in data_values.items():
        mask = img == rois[roi]
        new_img = new_img + mask * data_val

    final_img = sitk.GetImageFromArray(new_img, isVector=False)
    final_img.CopyInformation(itk_img)
    return final_img


# Include path to file.
def format_data(data_txt_file, lut_file):
    with open(lut_file) as lut:
        lut_data = lut.readlines()
    rois = []
    for x in lut_data:
        x = x.rstrip('\n')
        name = x.split(' ')[1]
        rois.append(name)

    with open(data_txt_file, 'r') as f:
        lines = f.readlines()

    data_values = {}
    for line in lines:
        if len(line) == 0:
            continue
        line = line.rstrip('\n')
        value = line.split(' ')[1]
        name = line.split(' ')[0]
        if name in rois:
            data_values[name] = float(value)
        else:
            continue
    return data_values


def run_customizer(output_path, lut_path, nii_path, fname, data_values={}, data_file_path=""):
    # make the folder structure for the outputs
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    # Take in the LUT and its values and make it into a dictionary
    with open(lut_path) as lut:
        f = lut.readlines()
    rois = {}
    for x in f:
        x = x.rstrip('\n')
        number = x.split(' ')[0]
        name = x.split(' ')[1]
        intensity = x.split(' ')[2]
        rois[name] = int(number)

    if len(data_values) != 0:
        # Use loaded data.
        # Go through the rois that we have values for and change their intensity values to the controllability value
        new_img = change_image(nii_path, data_values, rois)
        sitk.WriteImage(new_img, fileName=output_path + f'{fname}.nii.gz')
        return "Done."
    elif len(data_file_path) != 0:
        for root, dirs, files in os.walk(data_file_path):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                input_file = data_file_path + file
                print("Processing file ", input_file)

                # Take in the data values and make it into a dictionary
                with open(input_file) as control_values_file:
                    lines = control_values_file.readlines()
                data_values = {}

                for line in lines:
                    line = line.rstrip('\n')
                    value = line.split(' ')[1]
                    name = line.split(' ')[0]
                    data_values[name] = float(value)

                # Go through the rois that we have values for and change their intensity values to the controllability value
                new_img = change_image(nii_path, data_values, rois)
                fname = os.path.splitext(file)[0]
                sitk.WriteImage(new_img, fileName=output_path + f'{fname}.nii.gz')
        return "Done."
    else:
        return "ERROR. Missing data input."


def export_for_fsleyes(project, partition, fname, btype, reindex="top8"):
    """
    Create a new .nii file to import into FSLeyes. The .nii file is the image file that contains each node. Think of
    this function as modifying the *image* file of the brain atlas so that the image contains the partition information.
    :param project: project folder name to save to.
    :param partition: grouping of nodes in a network
    :param fname: output file name
    :param btype: a string, either "cortical", "subcortical", or "both"
    :param reindex: a string, either "top8", "+1", or something else. This is the encoding of the partition so that the
    .nii file output can have the same index information to create consistent coloring. "top8" will reindex the partition
    so that the modules are labelled from largest module size (1) to smallest module size (m). "+1" will take the
    module labels and add 1. Assuming module index is positive, this will remove any module with label 0, since 0 in
    FSLEyes is transparent. If neither strings are used, then there will be no re-labelling of the input partition.
    :return: None. Creates a fname.nii.gz file.
    """
    DATA_PATH = f'../../{project}/data/'
    FILE = fname + '.txt'
    # ---- PART 1
    if reindex == "top8":
        x = [p[0] for p in Counter(partition).most_common()]
        # RE-INDEX Community # from 1 = largest community, to smaller communities
        reindex_map = dict(zip(x, np.arange(1, len(Counter(partition)) + 1)))
        reindex_partition = [reindex_map[c] for c in partition]
    elif reindex == "+1":
        reindex_partition = [c + 1 for c in partition]
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
        formatted_data = format_data(data_txt_file, lut_file)
        run_customizer(output_path, lut_file, nii_file, fname=f'{FILE[:len(FILE) - 4]}_{btype}',
                               data_values=formatted_data)


def create_cortical_lut(partition, fname, lut_fname = 'Schaefer2018_1000Parcels_7Networks_order.lut'):
    """
    Creates a .lut file for the Schaefer2018 cortical Atlas. A LUT file is a Look Up Table which is used to visualize
    .nii files in FSLeyes. The look-up table will assign a value to each node in the brain atlas. This function
    will simply modify an existing brain atlas LUT file so that the value associated with each node corresponds to the
    module label of that node.
    The reasons you should use this function to create LUT files instead of "export_for_fsleyes" for visualizing community
    structure are:
        1. Faster - a LUT file is a much smaller file than an .nii image file. It's overlaying additional information
        rather than modifying a medical image file.
        2. Safer - with neuro-imaging files, there's specific orientation and format for each image. Modifying the image
        itself may accidentally change/didn't copy over certain meta info for the image. Modifying a LUT file doesn't
        have these problems.
        3. Consistent with others - Pretty sure this is the way neurologists use LUT file and the purpose of a LUT file.
        I've seen LUT files used in other labs for the exact same purpose (displaying community structure).
    Only use the "export_for_fsleyes" if you don't have a template LUT file for the brain atlas you're using.
    Note: Different atlases have different LUT formats. This function is built for the Schaefer2018 Atlas. May need to
    modify the inner for-loop for different Atlases. See "create_subcortical_lut" for an example.
    :param partition: community partition
    :param fname: output LUT file name
    :param lut_fname: path (including file name) of the template LUT file for a brain atlas
    :return: None. Creates a .lut in listed path.
    """
    color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 179, 102], 4: [0, 153, 77],
                 5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [0, 0, 0]}
    f = open(lut_fname, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")
        for i in range(1, 4):
            vec[i] = str(round(color_rgb[partition[l]][i - 1] / 255, 5))
        my_file_content += ' '.join(vec)
    with open(f"custom_lut/{fname}.lut", 'w') as output_file:
        output_file.write(my_file_content)


def create_subcortical_lut(partition, fname, lut_fname = 'Subcortical.nii.txt'):
    """
    Creates a .lut file for the Tian subcortex Atlas. Note: this function is built for a brain network that contains both
    cortical and subcortical nodes. The cortical nodes comes from the Schaefer2018 cortical Atlas while the subcortical
    nodes uses the Tian subcortex Atlas. The Schaefer2018 atlas contains 1000 nodes which is always the first 1000 entries
    in the partition. The subcortical nodes are all nodes after the first 1000 nodes. Depending on the atlas/combined
    atlas you use, modify the for-loop.
    :param partition: community partition
    :param fname: output .LUT file name
    :param lut_fname: template .lut file
    :return: None. Creates a .lut in listed path.
    """
    color_rgb = {0: [255, 51, 51], 1: [102, 179, 255], 2: [179, 102, 255], 3: [255, 179, 102], 4: [0, 153, 77],
                 5: [255, 204, 255], 6: [245, 211, 20], 7: [201, 0, 117], 8: [128, 128, 128], -1: [0, 0, 0]}
    f = open(lut_fname, "r")
    file_content = f.readlines()
    f.close()
    my_file_content = ""
    for l, line in enumerate(file_content):
        vec = line.split(" ")[:-1]
        my_file_content += f"{vec[0]} {round(color_rgb[partition[1000 + l]][0] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][1] / 255, 5)} " \
                           f"{round(color_rgb[partition[1000 + l]][2] / 255, 5)} " \
                           f"{vec[1]}\n"

    with open(f"custom_lut/{fname}_subcortex.lut", 'w') as output_file:
        output_file.write(my_file_content)
