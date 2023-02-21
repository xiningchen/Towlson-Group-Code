# Installing FSLeyes
- Follow installation instructions [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes#Install_as_part_of_FSL_.28recommended.29)
- Open computer terminal and type "fsleyes". Press enter. Application should open. 
- The bottom panel of the application has a "+" and "-" button. This is where you can add your atlas image files (.nii files). 
- For using .lut files, see instructions [here](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI)

# Visualizing in FSLeyes
There are two ways to visualize the community structure of a brain network in FSLeyes. 

1. Create a .nii file with the community information encoded in this image file. 
	- A .nii file is an image file that contains meta data on the neuro-imaging of the brain. 
2. Create a .lut file with the community information as the look up value. 
	- A .lut file is a Look Up Table that maps each node that exist in the correspoinding .nii file to some other value. 

Whenever you can, it is better to use method 2. This is because: 
1. Faster - a .lut file is a much smaller file than an .nii image file. It's overlaying additional information rather than modifying a medical image file.
2. Safer - with neuro-imaging files, there's metadata on the image orientation and format. Modifying the image itself may accidentally change/didn't copy over certain meta info for the image. Modifying a LUT file won't have these problems.
3. Consistent - Pretty sure this is the way neurologists use LUT file and the purpose of a LUT file. I've seen LUT files used in other labs for the exact same purpose (displaying community structure).

To start visualizing in FSLeyes, find the .nii and corresponding .lut files for your brain atlas. Sometimes an Atlas doesn't come with a .lut file. If you can find a .nii.txt file or some text file that contains the atlas node labels, then you can create a .lut file. 

# FSLeyes customizer code
In the provided file "fsleyes_customizer.py" there are 3 important functions: 
1. export_for_fsleyes(): This function modifies an existing .nii file to encode community information into the .nii image itself. 
2. create_cortical_lut(): This function modifies an existing .lut file to encode community information in the look up table. 
3. create_subcortical_lut(): Similar to the create_cortical_lut, it creates a .lut file with the community information in the look up table. The important difference is that this function creates the .lut file from a text file (.nii.txt). This code is an example of how you can use a .txt file provided by your Atlas to create a .lut file if the Atlas doesn't come with a .lut file. 


