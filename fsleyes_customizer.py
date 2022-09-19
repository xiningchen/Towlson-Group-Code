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

if __name__ == "__main__":
    # Path to LUT file
    LUT_PATH = 'Brain_Atlas/Cortical.nii.txt'
    # Path to NII file
    NII_PATH = 'Brain_Atlas/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
    # Path to Data file (used to replace image data with)
    DATA_FILE_PATH = 'data_to_plot/'
    # Output file path
    OUTPUT_PATH = 'fsleyes_custom/'
    for root, dirs, files in os.walk(DATA_FILE_PATH):
        for file in files:
            if not file.endswith('.txt'):
                continue
            input_file = DATA_FILE_PATH + file
            print("Processing file ", input_file)
            # make the folder structure for the outputs
            try:
                os.mkdir(OUTPUT_PATH)
            except FileExistsError:
                pass

            # Take in the LUT and its values and make it into a dictionary
            with open(LUT_PATH) as lut:
                f = lut.readlines()
            rois = {}
            for x in f:
                x = x.rstrip('\n')
                number = x.split(' ')[0]
                name = x.split(' ')[1]
                intensity = x.split(' ')[2]
                rois[name] = int(number)

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
            new_img = change_image(NII_PATH, data_values, rois)
            fname = os.path.splitext(file)[0]
            sitk.WriteImage(new_img, fileName=OUTPUT_PATH + f'{fname}.nii.gz')
