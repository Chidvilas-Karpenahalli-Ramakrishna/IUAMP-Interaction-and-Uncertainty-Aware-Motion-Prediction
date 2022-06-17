# Import other modules:
# -------------------------
from pathlib import Path
import json
import os

# Import project modules:
# ---------------------------
from data_preprocessor import preprocessor
from bev import bev
from ogm import ogm
from model import models, supporting_functions

# Import TensorFlow related:
# -------------------------------
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

# To prevent TensorFlow log outputs:
# ---------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# User-path definitions:
# -------------------------
nuscenes_data_input_path = "./data/sets/nuscenes"
nuscenes_data_version = 'v1.0-trainval'
obj_track_data_input_path = "./obj_track/tracking_result.json"
data_output_path = "./output"

"""
NOTE: 
1.) nuscenes_data_input_path: contains a list of paths for nuscenes meta-data. 
Once downloaded, the meta-data contains a folder v1.0-trainval, v1.0-mini etc. As 
per nuscenes guidelines, this directory must be copied to a folder of form
".../data/sets/nuscenes".
NuScenes meta-data doesn't contain map information. Please download
the map data from nuScenes and copy to the same folder.

2.) nuscenes_data_version = The version of the data downloaded from nuScenes 
Ex: v1.0-trainval, v1.0-mini etc.

3.) data_output_path: path to the directory where the generated files will be 
stored (this includes the gathered information file, formatted data files, 
bird's eye view (BEV) images) or Occupancy Grid Maps (OGMs).
"""

# User-defined parameters:
# ----------------------------- 
distance = (50, 50)
desired_ogm_size = (20, 20)
bev_fig_size = (1.5, 1.5)
bev_dpi = 200

"""
NOTE:
1.) distance: It is the distance that should be covered in BEV images and OGMs 
from the center. Distance is only in one direction. For Ex: 50 represents 50m in one direction,
which means the BEV image covers (100m, 100m) in total.
2.) desired_ogm_size: It is the desired number of row grids vs. column grids
3.) bev_fig_size = Represents the size of BEV images in inches
4.) bev_dpi = Represents "dots per inch" of BEV images and controls the quality
"""


## Tasks to perform:
## --------------------

# gather information from nuScenes meta-data:
gather_information_nuscenes = False
gather_information_obj_track = False

# format data and writing .csv files:
format_data_nuscenes = False
format_data_obj_track = False

# generate BEV images:
generate_bev_nuscenes = False
generate_bev_obj_track = False

# generate OGMs:
generate_ogm_nuscenes = False

# Train / test command:
train = False

"""
NOTE: all the relevant information for the preprocessing step will be taken from 
the files generated from gather_information step which is stored in the output 
directory specified by the user. The output of the data formatting step is also 
stored in the same output path. If you repeat any step, the files are merely 
replaced and won't cause any unexpected errors.

1.) gather_information: gathers the information from nuScenes meta-data to a single 
.json file for convenience. This is done to reduce the computational time as the 
meta-data from nuScenes contains a lot of irrelevant information for the present 
task and the tracked output of the object tracker can't be used directly. Choose 
'False' if .json file is already created.
NOTE: To run 'gather_information_obj_track', you must first run 
'gather_information_nuscenes' to generate the .json file which will be used as a
template to arrange the object tracker data.

2.) format_data: preprocess the information using the information gathered in 
the gather_information stage and generates pandas DataFrames and stores it as 
.csv file for each sample in each scene. This is done to speed up the process of 
bird's eye view image generation. This is also done to make sure that different 
datasets and the information provided by them can be brought to one data format.
Choose 'False' if the .csv files have already been generated for the given dataset.

3.) generate_bev: Generates BEV images for nuScenes and/or object-tracker 
based on user definition. Choose 'True' or 'False' accordingly.

4.) generate_ogm: Generates OGMs for nuScenes and/or object-tracker 
based on user definition. Choose 'True' or 'False' accordingly.

5.) train: Train the chosen model. Please define hyper-parameters and model 
parameters in 'experimentation_parameters.json' and 'model_parameters.json' 
respectively.
"""

# Steps to perfom based on user definition:
# ----------------------------------------------
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
        data_version=nuscenes_data_version, distance=distance, fig_size=bev_fig_size,
        dpi=bev_dpi)
    generator.generate_bev()

if generate_ogm_nuscenes:
    ogm_generator = ogm.NuscenesOGM(output_path=data_output_path,
                                    size_of_ogm=desired_ogm_size, distance=distance)
    ogm_generator.generate_ogm()

if train:
    try:
        # Load the experimentation parameters file defined by the user:
        # ----------------------------------------------------------------------
        parameters_path = os.path.join(os.path.dirname(__file__),  'experimentation_parameters.json')

        with open(Path(parameters_path)) as parameters_file:
            parameters = json.load(parameters_file)
            print("Successfully opened experimentation parameters file.")
            
            # Create train, validation and test split:
            # ------------------------------------------
            supporting_functions.data_divider(number_of_test_samples=parameters["number_of_test_samples"],
                                              validation_data_percentage=parameters["validation_data_split"],
                                              data_path=data_output_path, random_seed=parameters["random_seed"])

            """
            Counts the total number of steps that can be taken in 1 epoch. 
            This information is passed to TensorFlow.
            """
            train_steps = supporting_functions.steps_per_epoch_counter(input_data_path=data_output_path,
                                                                       time_steps=parameters["time_steps"],
                                                                       skip_steps=parameters["skip_steps"],
                                                                       batch_size=parameters["data_generator_batch_size"],
                                                                       random_seed=parameters["random_seed"],
                                                                       validate=False)

            val_steps = supporting_functions.steps_per_epoch_counter(input_data_path=data_output_path,
                                                                     time_steps=parameters["time_steps"],
                                                                     skip_steps=parameters["skip_steps"],
                                                                     batch_size=parameters["data_generator_batch_size"],
                                                                     random_seed=parameters["random_seed"],
                                                                     validate=True)

            # Choose which model to train on:
            # -----------------------------------
            model = models.model_options[parameters["which_model"]](
                desired_shape=tuple(parameters["desired_shape"]))

            # Compile the model:
            # ---------------------
            model.compile(loss=supporting_functions.cross_entropy_tversky_loss,
                          optimizer=Adam(learning_rate=parameters["learning_rate"]),
                          metrics=[MeanIoU(num_classes=2)])

            # Initialize the model:
            # ----------------------
            early_stopping, reduce_lr, tensorboard_callback, model_checkpoint_callback = \
                supporting_functions.model_initializer(log_path=data_output_path)

            # Train the model:
            # ------------------
            training_history = model.fit(supporting_functions.data_generator(
                desired_shape=tuple(parameters["desired_shape"]), input_data_path=data_output_path,
                time_steps=parameters["time_steps"], skip_steps=parameters["skip_steps"],
                batch_size=parameters["data_generator_batch_size"], random_seed=parameters["random_seed"],
                validate=False),
                epochs=parameters["epochs"], verbose=1, steps_per_epoch=train_steps,
                validation_data=supporting_functions.data_generator(
                    desired_shape=tuple(parameters["desired_shape"]), input_data_path=data_output_path,
                    time_steps=parameters["time_steps"], skip_steps=parameters["skip_steps"],
                    batch_size=parameters["data_generator_batch_size"],
                    random_seed=parameters["random_seed"], validate=True),
                validation_steps=val_steps,
                callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr, early_stopping])

    except FileNotFoundError:
        print("The experimentation parameters file was not found. Cancelling model training.")
