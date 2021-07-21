# IUAMP: Interaction and Uncertainty Aware Motion Prediction

<p align="center">
<img src="https://github.com/Chidvilas-Karpenahalli-Ramakrishna/MasterThesis/blob/main/additional_scripts/sample_33_14.png" width="500" height="500">
</p>


**Topic:** Uncertainty aware observation of surrounding traffic agents for interaction aware motion prediction models.

The aim of the present work is to observe the surrounding traffic agents and predict their future motion along with an uncertainty measure. Both the aleatoric and epistemic uncertainty is modelled. The input to the network is a bird's eye view (BEV) image of the entire scene and the expected output is an occupancy grid map (OGM) of the future scene. Uncertainty propagation from object tracker module to the motion prediction module is also being studied by propagating confidence measure as encoded Saturation value in the HSV space.


## Tasks:
- [x] Preprocess data
- [x] Generate BEV images (inputs to the network - x)
- [x] Generate OGM (targets to the network - y)
- [ ] Build uncertainty aware prediction network (ConvLSTM)
- [ ] Train, test and optimize


## Project Files Overview:
- **`main.py`** (single-point access for user definitions)
- **`nuscenes`** (nuscenes devkit with modified `map_api.py`) 
- **`data_preprocessor`** (package to preprocess the data) 
  - **`preprocessor.py`** (main preprocessing file that calls `info_gather.py` and `format_data.py`)
  - **`info_gather.py`** (to perform the information gathering step)
  - **`format_data.py`** (to perform the data formatting step)
- **`bev`** (package to generate BEV images)
  - **`bev.py`** (main file to generate BEV images)
  - **`colour_code_nuscenes.json`** (contains the object class-wise colour coding information for nuscenes data)  
  - **`colour_code_obj_track.json`** (contains the object class-wise colour coding information for object tracker data)
- **`ogm`** (package to generate OGMs)
  - **`ogm.py`** (main file to generate OGMs)
  - **`ogm_objects_nuscenes.json`** (contains the objects of interest for ogm generation. This much match the objects considered in BEVs)
- **`additional scripts`** (contains additional scripts for visualization and understanding)
  - **`bev_principle.ipynb`** (notebook illustrating the principle used to generate BEV images by drawing bounding boxes)
  - **`ogm_principle.ipynb`** (notebook illustrating the principle used to generate OGMs from the formatted data)

## Project Overview:
- [Project setup](#project-setup)
- [Preprocess data](#preprocess-data)
  - [Gather information](#gather-information) 
  - [Format data](#format-data)
- [Generate BEV](#generate-bev)
- [Generate OGM](#generate-ogm)

## Project setup:
1. Create a data-input folder of the form `../data/sets/nuscenes`
2. Download the trainval v1.0 meta-data and v1.3 map-expansion from the [nuScenes download page](https://www.nuscenes.org/download)
3. Extract the contents to the folder `../data/sets/nuscenes`. This becomes the data input folder.
4. Rename the object tracker output to `tracking_result.json` and copy to `../data/object_track/tracking_result.json`.
5. Please provide these paths as user definitions for the data preprocessing step.
6. Create an output directory `../output`. And pass this as the output path during data preprocessing and BEV generation step. Please note that the output of all steps will be stored in the same output directory. The code creates separate directories for each step.

## Preprocess data:
Preprocessing step gathers the data necessary for the generation of bird's eye view (BEV) images and occupancy grid maps (OGMs). It is a two step process for both nuScenes meta-data and object-tracker data. The first step is [information gathering](#gather-information) step and the second step is [data formatting](#format-data) step.   

### Gather information:
Information gathering is the process of acquiring required tokens like location information, ego-pose information, scene information and the respective samples information from nuSenes meta-data and also reformatting the output of the [Center-based 3D object tracker](https://github.com/tianweiy/CenterPoint). A single `.json` file is created with all the relevant information. This enables faster data formatting and the BEV generation. The output file is of `.json` format and is stored in the output directory.

### Format data:
Data formatting step gathers the bounding box information in a tabular format as Pandas DataFrames and generates `.csv` files for each sample in each scene. The output is stored scene-wise in `..output/nuscenes_formatted_data` for nuScenes and `..output/obj_track_formatted_data` for object tracker data. The center (X,Y) of objects, (X-anchor, Y-anchor) and heading-angle (yaw angle) information is calculated for generating BEV images and other corner coordinates of the bounding box are calculated for generating OGMs. The object tracker formatted data contains the same information along with an additional column of confidence measure for each tracked object.

## Generate BEV:
BEV or bird's eye view images are generated for each scene in the nuScenes dataset and for the object tracker output as per user definition. BEV generation step acquires required information from information files and data from formatted data folders. So, it is important to perform the steps in the given order. BEV images are colour coded in HSV space. Where the Hue value encodes the object-class information, the Saturation space is used to encode confidence measure which is a part of aleatoric uncertainty (1.0 in case of nuScenes ground truth data and actual confidence measure in case of object tracker). The Value space can be used to encode velocity (At present kept at maximum value of 1). The BEV images have an underlying map of the location with highlighted drivable-area. The BEV images cover an area of 80m in all direction from the EGO vehicle at the center.

## Generate OGM:
OGM or occupancy grid maps are a sparse representation of the scene where the scene is divided into grid cells each having a binary value of 0 (empty) or 1 (occupied). OGMs are generated and stored as 2D-numpy arrays. The model is designed to take BEVs as input and predict the future OGMs with an uncertainty measure.
