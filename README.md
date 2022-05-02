# Group code for starting programming in python related to brain data
See 'example_demo.ipynb' for examples of using functions in python files

### Input / output (data_io.py)
* Functions for reading in excel data of brain scans or MATLAB data
* Exporting brain connectome in BrainNet Viewer format (.node and .edge files)

### Brain connectomes (WIP - brain_network.py)
* Function for averaging brain connectomes
* Function for getting basic network statistics (WIP)

### Functional Brain Community Detection with bctpy
* Function for finding a good resolution parameter value for Louvain detection
* Function for Louvain community detection with finer grain
* Function for plotting partition (WIP)

### Resources/References 
* [Brain Connectivity Toolbox (Python)](https://pypi.org/project/bctpy/)
* [Community Louvain 2021 - MATLAB](https://drive.google.com/drive/folders/1P32DAUy1AFEn7biMomD0v8j373byRAOq)
* [ARC Cluster Guide](https://github.com/mariamasoliver/connect_to_ARC/blob/main/Guide.md) by Maria Masoliver and Thomas Newton. 
	- If the software you need isn't available in the modules you can email:
support@hpc.ucalgary.ca and they can help get it installed.
	- You may also be able to install it on your own home directory if the source code is all in python. Self-install by uploading all the source code to your ARC account, usually in the same folder that the calling code is located. 
	- The [Compute Canada wiki] has some info on installing software that might be useful.(https://docs.computecanada.ca/wiki/Installing_software_in_your_home_directory
	- [Apply for ARC](https://rcs.ucalgary.ca/How_to_get_an_account)
