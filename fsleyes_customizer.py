import os
import SimpleITK as sitk
# These two global variables need to be changed.
# The LUT_PATH needs to be changed to the path to the Look-up table (LUT) file that maps each region
# to its intensity value on the atlas.
# The CONTROL_VALUES_FILE_PATH needs to be changed to the path to the file containing the values needed
# to visualized (in this case controllability values). This text file needs to be formatted as rows,
# with each row containing a region name, followed by a space, then the value.
# e.g.: Precentral_R 55

def change_image(IMAGE_PATH, data_values, rois):
    itk_img = sitk.ReadImage(IMAGE_PATH)
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
