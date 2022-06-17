# IUAMP: Interaction and Uncertainty Aware Motion Prediction

<table>
  <tr>
    <td align="center">Object-tracker BEV image</td>
    <td align="center">Mean prediction (per grid-cell)</td>
    <td align="center">Uncertainty prediction (per grid-cell)</td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/Chidvilas-Karpenahalli-Ramakrishna/MasterThesis/blob/main/BEV.gif" width="350" alt="BEV" title="BEV"/></td>
    <td align="center"><img src="https://github.com/Chidvilas-Karpenahalli-Ramakrishna/MasterThesis/blob/main/Mean prediction.gif" width="350"/></td>
    <td align="center"><img src="https://github.com/Chidvilas-Karpenahalli-Ramakrishna/MasterThesis/blob/main/Uncertainty in prediction.gif" width="350"/></td>
  </tr>
 </table>

**Master's thesis topic:** Uncertainty Aware Observation of Surrounding Traffic Agents for Interaction Aware Motion Prediction Models.

The aim of the master's thesis is to observe the surrounding traffic agents and predict their future motion along with an uncertainty measure in a challenging heterogenous urban environment. The epistemic uncertainty which represents a "lack of knowledge" on the part of the neural network is modelled here. The input to the network is a sequence of color-coded bird's eye view (BEV) images of the entire scene and the expected output is a sequence of occupancy grid maps (OGM) of the future scene. So, the problem can be stated as a Seq2Seq problem. Uncertainty propagation from object tracker module to the motion prediction module is studied by propagating confidence measure as encoded Saturation value in the HSV space.

**Institution:**<br>

The master's thesis was written at the "Center of Automotive Research on Integrated Safety Systems and Measurement Area - Institute of Safety in Future Mobility (C-ISAFE)" located on the campus of Technische Hochschule Ingolstadt (THI), Germany. The thesis was supervised by Prof. Dr. Christian Birkner, Prof. Ondrej Vaculin, Ph.D. and Leon Tolksdorf, M.Sc.<br>

**Note:** The master's thesis is available for public access at THI library.


## Master's thesis tasks:
- [x] Preprocess and format nuScenes data
- [x] Generate BEV images (inputs to the network - x)
- [x] Generate OGM (targets to the network - y)
- [x] Build Deterministic and Uncertainty-aware motion prediction networks
- [x] Train, test and optimize the networks
- [x] Check the performance on object-tracker data 
- [x] Document the work (thesis report, GitHub update, presentation) 


## Project Files Overview:
- **`bev`** (Package to generate BEV images)
  - **`bev.py`** (Main file to generate BEV images)
  - **`colour_code_nuscenes.json`** (Contains the object class-wise colour coding information for nuscenes data)  
  - **`colour_code_obj_track.json`** (Contains the object class-wise colour coding information for object tracker data)

- **`data_preprocessor`** (Package to preprocess the data) 
  - **`preprocessor.py`** (Main preprocessing file that calls `info_gather.py` and `format_data.py`)
  - **`info_gather.py`** (Python file to perform the "information gathering" step)
  - **`format_data.py`** (Python file to perform the "data formatting" step)
  
- **`model`** (Package containing the neural networks used in the present thesis)
  - **`model_parameters.json`** (Contains hypterparameters for the models that can be defined by the users)
  - **`models.py`** (Python file containing models used in the thesis. The models are built using TensorFlow and Keras)
  - **`supporting_functions.py`** (Python file containing supporting methods. This includes attention mechanisms, novel Combi-loss, permanent dropout etc.)

- **`nuscenes`** (Original nuscenes devkit with modified `map_api.py`) 

- **`ogm`** (Package to generate OGMs)
  - **`ogm.py`** (main file to generate OGMs)
  - **`ogm_objects_nuscenes.json`** (contains the objects of interest for ogm generation. This must match the objects considered in BEVs)

- **`utils`** (Contains additional scripts for visualization and understanding)
  - **`bev_principle.ipynb`** (notebook illustrating the principle used to generate BEV images by drawing bounding boxes)
  - **`ogm_principle.ipynb`** (notebook illustrating the principle used to generate OGMs from the formatted data)

- **`experimentation_parameters.json`** (Contains training parameters that can be defined by the user)

- **`main.py`** (A single point main access point for users. You can define what you want to do with the scripts here.)


## Project Overview:
- [Project setup](#project-setup)
- [Preprocess data](#preprocess-data)
  - [Gather information](#gather-information) 
  - [Format data](#format-data)
- [Generate BEV](#generate-bev)
- [Generate OGM](#generate-ogm)
- [Train and test](#train-and-test)


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
Information gathering is the process of acquiring required tokens like location information, ego-pose information, scene information and the respective samples information from nuSenes meta-data and also reformatting the output of the [Center-based 3D object tracker](https://github.com/tianweiy/CenterPoint). A single `.json` file is created with all the relevant information. This enables faster data formatting and BEV generation. The output file is of `.json` format and is stored in the output directory.

### Format data:
Data formatting step gathers the bounding box information in a tabular format (Pandas DataFrames) and generates `.csv` files for each sample in each scene. The output is stored scene-wise in `..output/nuscenes_formatted_data` for nuScenes and `..output/obj_track_formatted_data` for object tracker data. The center (X,Y) of objects, (X-anchor, Y-anchor) and heading-angle (yaw angle) information is calculated for generating BEV images and other corner coordinates of the bounding box are calculated for generating OGMs. The object tracker formatted data contains the same information along with an additional column of confidence measure for each tracked object.

## Generate BEV:
BEV or bird's eye view images are generated for each scene in the nuScenes dataset and for the object tracker output as per user definitions. BEV generation step acquires required information from information files and data from formatted data folders. So, it is important to perform the steps in the given order. BEV images are colour coded in HSV space. Where the Hue value encodes the object-class information, the Saturation space is used to encode confidence measure which is a part of aleatoric uncertainty (1.0 in case of nuScenes ground truth data and actual confidence measure in case of object tracker). The Value space can be used to encode velocity (At present kept at maximum value of 1). The BEV images have an underlying map of the location with highlighted drivable-area. 

## Generate OGM:
OGM or occupancy grid maps are a sparse representation of the scene where the scene is divided into grid cells. Each grid cell has a binary value of 0 (empty) or 1 (occupied). OGMs are generated and stored as 2D-numpy arrays. The IUAMP models are designed to take a sequence of BEVs as input and predict the future OGM sequences with an uncertainty measure.

## Train and test:
In the current thesis, we train motion prediction models on 2 second and 3 second horions (both observation and prediction horizons are kept same). Our motion prediction model is of an encoder-decoder architecture. Where the encoder consists of a feature extractor comprised of convolutional layers. This is followed by a series of ConvLSTM layers which are meant for spatio-temporal learning. This is followed by a sequence self-attention mechanism (multiplicative attention) that encodes the output of the ConvLSTM layers. The decoder is a simple fully-connected layer which outputs a sequence of OGMs. The output of the fully-connected layer is reshped to match the shape of the target OGMs. Monte Carlo Dropout (MC Dropout) is used to estimate epistemic uncertainty in our motion prediction model. In the current work, we also introduce a novel Combi-loss function (Binary Crossentropy + Tversky loss) which is used to train our models. Testing is performed on a nuScenes testset and on object-tracker data.<br>

The performance of our model can be visualized in the GIF image above, for nuScenes scene-0015.
