# Required build-in modules:
# -----------------------------
import json
import os
import gc
from pathlib import Path

# Required project modules:
# -----------------------------
from data_preprocessor import info_gather, format_data


# Info-gathering class for nuScenes data:
# ---------------------------------------------
class NuScenesInfoGather:
    """
    Class to gather information from the meta-data provided by nuScenes and
    convert to a single .json file to effectively handle memory.
    """

    def __init__(self, input_path, output_path, version):
        """
        :param input_path: Path where nuScenes meta-data is stored

        :param ouput_path: Path where the output files are stored

        :param version: The version of nuScenes data dowloaded
        """
        self.data_version = version
        self.input_path = os.sep.join([input_path, version])
        self.output_path = output_path
        self.nuscenes_info = []
    

    def get_info(self):
        """
        This method calls the info_gather.py module which has the methods that open
        the relevant nuScenes meta-data .json files and gathers the necessary
        tokens needed for the task of formatting data and bird's eye view image
        generation task.
        """
        print("Started info gathering for nuscenes data")
        try:
            input_path = Path(self.input_path).resolve(strict=True)
            nuscenes_info_getter = info_gather.NuScenesInfo(input_path)
            self.nuscenes_info = nuscenes_info_getter.scene_list_getter()
            self.nuscenes_info = nuscenes_info_getter.\
                sample_token_getter(self.nuscenes_info)
            self.nuscenes_info = nuscenes_info_getter.\
                location_getter(self.nuscenes_info)
            self.nuscenes_info = nuscenes_info_getter.\
                ego_pose_token_getter(self.nuscenes_info)

            with open(Path(os.sep.join([self.output_path,
                                        f'{self.data_version}.json'])), 'w') as outfile:
                json.dump(self.nuscenes_info, outfile)

            del nuscenes_info_getter
            gc.collect()
            print("Info gathering for nuscenes data successful.")

        except FileNotFoundError:
            print(f"The path {self.input_path} does not contain the input data")


# Info gathering class for object tracker data:
# ---------------------------------------------------
class ObjTrackInfoGather:
    """
    Class to gather information from the output of the object tracker and combine
    it with the nuScenes scenes and samples information to keep the tracked data
    in a structured format.
    """

    def __init__(self, nuscenes_info_path, obj_track_input_path, output_path,
                 data_version):
        """
        :param nuscenes_info_path: The path to the directory where info files is stored.
        
        :param obj_track_input_path: The path to the directory where the .json file of the 
        object tracker is stored

        :param ouput_path: The path of the directory where the output files will be saved.
        """
        self.nuscenes_info_path = nuscenes_info_path
        self.obj_track_input_path = obj_track_input_path
        self.output_path = output_path
        self.data_version = data_version


    def get_info(self):
        obj_track_info_getter = info_gather.ObjTrackInfo(
            self.nuscenes_info_path, self.obj_track_input_path, self.data_version)
        __obj_track_info = obj_track_info_getter.get_scene_sample_info()

        with open(Path(os.sep.join([self.output_path,
                                    'obj_track_info.json'])), 'w') as outfile:
            json.dump(__obj_track_info, outfile)


# Data formatting class for nuScenes data:
# ----------------------------------------------
class NuScenesDataFormatter:
    """
    Class contains methods to convert the nuScenes data to a tabular format stored
    as .csv files for each scene in an organised manner. This serves three purposes:
    1.) Bird's eye view (BEV) image generation needs heading angles and anchor points
    which is not readily available in a usable format. Data formatting allows us to
    perform these steps before generating BEV image.
    1.) The bird's eye view generation is faster as only the formatted .csv files are
    accessed and there is more flexibility to modify BEV images by modifying the
    data formatting step if additional information is needed.
    2.) If in future other datasets like Waymo or KITTI are used, it makes it easier
    to bring all the datasets to one format so that the remaining code can be reused.
    """

    def __init__(self, input_path, output_path, version, gathered_info_path):
        """
        :param input_path: The path to the directory where nuScenes meta-data is stored

        :param output_path: The path of the directory where the output files will be saved

        :param version: The version of the nuScenes dataset download.

        :param gathered_info_path: The path to the directory where the info files are stored 
        from the previous stages
        """
        self.data_version = version
        self.input_path = input_path
        self.output_path = output_path
        self.gathered_info_path = gathered_info_path


    def create_data_frame(self):
        """
        Method calls the module 'format_data.py' which contains methods that gather
        the meta-data, get heading angle and anchor points.
        """
        print("Started formatting data for nuscenes")
        formatter = format_data.NuScenesDataFormat(input_path=self.input_path,
                                                   output_path=self.output_path,
                                                   version=self.data_version,
                                                   gathered_info_path=self.gathered_info_path)
        formatter.read_files()
        formatter.data_formatter()


# Data formatting class for object tracker:
# -----------------------------------------------
class ObjTrackDataFormatter:
    """
    Formats the data similar to NuScenes where the formatted data is stored
    scene-wise within which the tracked samples are present.
    """

    def __init__(self, input_path, output_path, version, gathered_info_path):
        """
        :param input_path: The path to the directory where nuScenes meta-data is stored

        :param output_path: The path of the directory where the output files will be saved

        :param version: The version of the nuScenes dataset download.

        :param gathered_info_path: The path to the directory where the info files are stored 
        from the previous stages
        """
        self.data_version = version
        self.input_path = input_path
        self.output_path = output_path
        self.gathered_info_path = gathered_info_path


    def create_data_frame(self):
        print("Started formatting data for object tracker")
        formatter = format_data.ObjTrackDataFormat(input_path=self.input_path,
                                                   output_path=self.output_path,
                                                   version=self.data_version,
                                                   gathered_info_path=self.gathered_info_path)
        formatter.read_files()
        formatter.data_formatter()
