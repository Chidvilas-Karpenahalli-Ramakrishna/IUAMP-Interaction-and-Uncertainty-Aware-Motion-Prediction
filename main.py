from data_preprocessor import preprocessor
from bev import bev
from ogm import ogm

# User-path definitions:
"""
Please change the paths as needed. 
1.) nuscenes_data_input_path: contains a list of paths for nuscenes meta-data. 
Once downloaded, the meta-data contains a folder v1.0-trainval, v1.0-mini etc. As 
per nuscenes guidelines, this directory must be copied to a folder of form
".../data/sets/nuscenes".
NOTE: NuScenes meta-data doesn't contain map information. Please download
the map data from nuScenes and copy to the same folder.
2.) nuscenes_data_version = The version of the data downloaded from nuScenes 
Ex: v1.0-trainval, v1.0-mini etc.
3.) data_output_path: path to the directory where the generated files will be 
stored (this includes the gathered information file, formatted data files and the 
bird's eye view (BEV) images).
"""

nuscenes_data_input_path = '/home/chid/Master thesis/IAUMP/data/sets/nuscenes'

nuscenes_data_version = 'v1.0-trainval'

obj_track_data_input_path = '/home/chid/Master thesis/IAUMP/' \
                            'data/obj_track/tracking_result.json'

data_output_path = '/home/chid/Master thesis/IAUMP/output'

# Defining additional parameters:
'''
1.) desired_ogm_size: It is the desired number of row grids vs. column grids
2.) distance: It is the distance that should be covered in BEV images and OGMs 
from the center.
Note: distance is only in one direction. For Ex: 80 represents 80m in one direction,
which means the BEV image covers (160m, 160m)
'''
distance = (80, 80)
bev_fig_size = (1.5, 1.5)
bev_dpi = 200
desired_ogm_size = (10, 10)


# Choose what steps you wish to perform:
"""
NOTE: all the relevant information for the preprocessing step will be taken from 
the files generated from gather_information step which is stored in the output 
directory specified by the user. The output of the data formatting step is also 
stored in the same output path. If you repeat any step, the files are merely 
replaced and won't cause any unexpected errors.

1.) gather_information: gathers the information from the meta-data to a single 
.json file for convenience. This is done to reduce the computational time as the 
meta-data from nuScenes contains a lot of irrelevant information for the present 
task and the tracked output of the object tracker can't be used directly. Choose 
'False' if .json file already exists.
NOTE: To run 'gather_information_obj_track', you must first run 
'gather_information_nuscenes' to generate the .json file which will be used as a
template to arrange the object tracker data.

2.) format_data: preprocess the information using the information gathered in 
the gather_information stage and generates pandas DataFrames and stores it as 
.csv file for each sample in each scene. This is done to speed up the process of 
bird's eye view image generation. This is also done to make sure that different 
datasets and the information provided them can be brought to one data format.
Choose 'False' if the .csv files are already generated for the given dataset.

3.) generate_bev: generates BEV images for nuScenes and/or object-tracker 
based on user definition. Choose True or False accordingly.
"""

# gather information from nuScenes meta-data:
gather_information_nuscenes = False
gather_information_obj_track = False

# format data and writing .csv files
format_data_nuscenes = False
format_data_obj_track = False

# generate BEV images:
generate_bev_nuscenes = False
generate_bev_obj_track = False

# generate OGMs:
generate_ogm_nuscenes = True
generate_ogm_obj_track = False


# Logic based on user definition:

if gather_information_nuscenes:
    info_gatherer = preprocessor.NuScenesInfoGather(
        input_path=nuscenes_data_input_path, output_path=data_output_path,
        version=nuscenes_data_version)
    info_gatherer.get_info()

if gather_information_obj_track:
    info_gatherer = preprocessor.ObjTrackInfoGather(
        nuscenes_info_path=data_output_path,
        obj_track_input_path=obj_track_data_input_path,
        output_path=data_output_path, data_version=nuscenes_data_version)
    info_gatherer.get_info()

if format_data_nuscenes:
    data_formatter = preprocessor.NuScenesDataFormatter(
        input_path=nuscenes_data_input_path, output_path=data_output_path,
        version=nuscenes_data_version, gathered_info_path=data_output_path)
    data_formatter.create_data_frame()

if format_data_obj_track:
    data_formatter = preprocessor.ObjTrackDataFormatter(
        input_path=nuscenes_data_input_path, output_path=data_output_path,
        version=nuscenes_data_version, gathered_info_path=data_output_path)
    data_formatter.create_data_frame()

if generate_bev_nuscenes:
    generator = bev.NuScenesBEV(
        nuscenes_data_path=nuscenes_data_input_path, out_path=data_output_path,
        data_version=nuscenes_data_version, distance=distance, fig_size=bev_fig_size,
        dpi=bev_dpi)
    generator.generate_bev()

if generate_bev_obj_track:
    generator = bev.ObjTrackBEV(
        nuscenes_data_path=nuscenes_data_input_path, out_path=data_output_path,
        data_version=nuscenes_data_version)
    generator.generate_bev()

if generate_ogm_nuscenes:
    ogm_generator = ogm.NuscenesOGM(output_path=data_output_path,
                                    size_of_ogm=desired_ogm_size, distance=distance)
    ogm_generator.generate_ogm()
