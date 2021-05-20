# Built-in modules:
import os
import shutil
import json
import gc
from pathlib import Path

# Other dependencies:
from tqdm import tqdm
import pandas as pd
import colorsys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Importing nuScenes modules:
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap


class NuScenesBEV:
    """
    The class reads the information from the formatted data .csv files and generates
    BEV images which contain the position (x, y), size, heading direction, underlying
    map with drivable area and colour coded class information.
    The colour coding is defined in colour_code_nuscenes.json file in the current directory.
    The colour coding is done in HSV space using Hue as the dimension to code
    object class, saturation is kept at max here and is used to code confidence
    measure. For ground truth the confidence will be 100% and hence, saturation=1.
    The Value space can be used to code velocity and at present not considered and
    kept at 1.
    """
    def __init__(self, nuscenes_data_path, out_path, data_version):
        self.__input_path = nuscenes_data_path
        self.__nuscenes_info_path = os.sep.join([out_path, data_version])
        self.__out_path = out_path
        self.__formatted_data_path = os.sep.join(
            [out_path, 'nuscenes_formatted_data'])

    def plot_bev(self, plotting_data, location, bev_save_path):
        """
        The method plots the BEV. It reads the map information provided by
        nuScenes and plots the objects on the map and also crops the map based on
        user-definition. The BEV images are saved scene-wise in separate folders.
        """
        try:
            __file_name = 'colour_code_nuscenes.json'
            __present_file = os.path.abspath(__file__)
            __present_dir = os.path.dirname(__present_file)
            __code_file = os.path.join(__present_dir, __file_name)

            with open(Path(__code_file)) as __code_file:
                __code = json.load(__code_file)
                __nusc_map = NuScenesMap(dataroot=self.__input_path, map_name=location)
                __bitmap = BitMap(__nusc_map.dataroot, __nusc_map.map_name, 'basemap')
                __fig, __ax = __nusc_map.render_layers(['drivable_area'], bitmap=__bitmap)

                for i in range(0, len(plotting_data)):
                    if plotting_data['category'][i] in __code:
                        __x = plotting_data['x_anchor'][i]
                        __y = plotting_data['y_anchor'][i]

                        """
                        hsv_to_rgb: converts the HSV space to RGB for generation of BEV.
                        As seen the first value is object class and is Hue. The second value 
                        is the Saturation which is 100% in case of ground truth. The 
                        third value is Value and is kept at 1.
                        """
                        __colour = colorsys.hsv_to_rgb(
                            __code[plotting_data['category'][i]]/360, 1, 1)

                        __ax.add_patch(matplotlib.patches.Rectangle(xy=(__x, __y),
                                                                    width=plotting_data['width'][i],
                                                                    height=plotting_data['height'][i],
                                                                    angle=plotting_data['heading_angle'][i],
                                                                    color=__colour))

                """
                Crop the area as desired. Currently 80m is cropped from the center in
                all four directions.
                """
                plt.ylim(bottom=(plotting_data['y'][0])-80, top=(plotting_data['y'][0])+80)
                plt.xlim(left=(plotting_data['x'][0])-80, right=(plotting_data['x'][0])+80)
                plt.grid('off')
                plt.axis('off')
                plt.savefig(f'{bev_save_path}.png')
                plt.close()

            """
            Free some memory. Unreachable objects from the NuScenes module can 
            cause memory issues and trigger:
            exit code 137 (interrupted by signal 9: SIGKILL) on Pycharm
            """
            del __code, __nusc_map, __bitmap, __fig, __ax
            gc.collect()

        except FileNotFoundError:
            print("Please make sure the colour_code_nuscenes.json file is present in the directory.")

    def generate_bev(self):
        """
        The method is the main method that calls other methods in the class for the
        task fo BEV generation.
        """
        __scenes_list = []
        __scene_names = []
        """
        The formatted data which is stored scene-wise is read as below for the task
        of BEV generation.
        """
        for path, subdir, files in os.walk(self.__formatted_data_path):
            for name in sorted(subdir):
                __scene_names.append(name)
                __scenes = os.path.join(path, name)
                __scenes_list.append(__scenes)

        # creating an output path to store BEV images
        __nuscenes_bev_path = os.sep.join([self.__out_path, 'bev_nuscenes'])
        if os.path.exists(__nuscenes_bev_path):
            shutil.rmtree(__nuscenes_bev_path)
        os.makedirs(__nuscenes_bev_path)

        try:
            # Load info file to access the location information
            with open(Path(f"{self.__nuscenes_info_path}.json")) as nuscenes_info_file:
                __nuscenes_info = json.load(nuscenes_info_file)
                print("Successfully opened nuScenes info file.")

                for i in tqdm(range(0, len(__scenes_list)),
                              desc='Creating BEV images for nuScenes data', colour='green'):

                    # Create a scene path to store the scene BEVs
                    __bev_scene_path = os.sep.join([self.__out_path, 'bev_nuscenes',
                                                    __scene_names[i]])
                    if os.path.exists(__bev_scene_path):
                        shutil.rmtree(__bev_scene_path)
                    os.makedirs(__bev_scene_path)

                    for x in range(0, len(__nuscenes_info)):
                        if __nuscenes_info[x]['name'] == __scene_names[i]:
                            __scene_location = __nuscenes_info[x]['location']
                            break

                    __samples_list = sorted(os.listdir(__scenes_list[i]),
                                            key=lambda item: int(item.split('_')[0]))
                    for each_sample in __samples_list:
                        __sample_path = os.sep.join([__scenes_list[i], each_sample])
                        __data = pd.read_csv(__sample_path)
                        __path_to_save = os.sep.join([__bev_scene_path,
                                                      each_sample.split('.')[0]])
                        self.plot_bev(plotting_data=__data, location=__scene_location,
                                      bev_save_path=__path_to_save)

        except (FileNotFoundError, IsADirectoryError):
            print("NuScenes info file not found. Please provide proper directory or "
                  "run info gather step before attempting to generate BEV images.")


class ObjTrackBEV:
    """
    The class reads the information from the formatted data .csv files and generates
    BEV images which contain the position (x, y), size, heading direction, underlying
    map with drivable area and colour coded class information.
    The colour coding is defined in colour_code_nuscenes.json file in the current directory.
    The colour coding is done in HSV space using Hue as the dimension to code
    object class, saturation is kept at max and Value space is used to code confidence
    measure. The object tracker provides a confidence measure which is available as
    tracking score and the Saturation space is used to code the tracking score.
    Value is kept at 1 with the possibility to code velocity information.
    NOTE: The object tracker generalises many object classes Ex: vehicle.car,
    vehicle.emergency.police are detected are mere car etc. And based on this, the
    colour coding is done to match with the ground truth.
    """
    def __init__(self, nuscenes_data_path, out_path, data_version):
        self.__input_path = nuscenes_data_path
        self.__nuscenes_info_path = os.sep.join([out_path, data_version])
        self.__out_path = out_path
        self.__formatted_data_path = os.sep.join(
            [out_path, 'obj_track_formatted_data'])

    def plot_bev(self, plotting_data, location, bev_save_path):
        """
        The method plots the BEV. It reads the map information provided by
        nuScenes and plots the objects on the map and also crops the map based on
        user-definition. The BEV images are saved scene-wise in separate folders.
        """
        try:
            __file_name = 'colour_code_obj_track.json'
            __present_file = os.path.abspath(__file__)
            __present_dir = os.path.dirname(__present_file)
            __code_file = os.path.join(__present_dir, __file_name)

            with open(Path(__code_file)) as __code_file:
                __code = json.load(__code_file)
                __nusc_map = NuScenesMap(dataroot=self.__input_path, map_name=location)
                __bitmap = BitMap(__nusc_map.dataroot, __nusc_map.map_name, 'basemap')
                __fig, __ax = __nusc_map.render_layers(['drivable_area'], bitmap=__bitmap)

                for i in range(0, len(plotting_data)):
                    if plotting_data['category'][i] in __code:
                        __x = plotting_data['x_anchor'][i]
                        __y = plotting_data['y_anchor'][i]

                        """
                        hsv_to_rgb: converts the HSV space to RGB for generation of BEV.
                        As seen the first value is object class and represents the Hue value 
                        and the second value is the tracking score which is the confidence 
                        measure and represents the Saturation. The third value is the Value
                        and is kept at 1. Velocity can be coded here.
                        """
                        __colour = colorsys.hsv_to_rgb(
                            __code[plotting_data['category'][i]]/360,
                            plotting_data['tracking_score'][i], 1)

                        __ax.add_patch(matplotlib.patches.Rectangle(xy=(__x, __y),
                                                                    width=plotting_data['width'][i],
                                                                    height=plotting_data['height'][i],
                                                                    angle=plotting_data['heading_angle'][i],
                                                                    color=__colour))

                """
                Crop the area as desired. Currently 80m is cropped from the center in
                all four directions.
                """
                plt.ylim(bottom=(plotting_data['y'][0])-80, top=(plotting_data['y'][0])+80)
                plt.xlim(left=(plotting_data['x'][0])-80, right=(plotting_data['x'][0])+80)
                plt.grid('off')
                plt.axis('off')
                plt.savefig(f'{bev_save_path}.png')
                plt.close()

            """
            Free some memory. Unreachable objects from the NuScenes module can 
            cause memory issues and trigger:
            exit code 137 (interrupted by signal 9: SIGKILL) on Pycharm
            """
            del __code, __nusc_map, __bitmap, __fig, __ax
            gc.collect()

        except FileNotFoundError:
            print("Please make sure the colour_code_nuscenes.json file is present in the directory.")

    def generate_bev(self):
        """
        The method is the main method that calls other methods in the class for the
        task fo BEV generation.
        """
        __scenes_list = []
        __scene_names = []
        """
        The formatted data which is stored scene-wise is read as below for the task
        of BEV generation.
        """
        for path, subdir, files in os.walk(self.__formatted_data_path):
            for name in sorted(subdir):
                __scene_names.append(name)
                __scenes = os.path.join(path, name)
                __scenes_list.append(__scenes)

        # creating an output path to store BEV images
        __nuscenes_bev_path = os.sep.join([self.__out_path, 'bev_obj_track'])
        if os.path.exists(__nuscenes_bev_path):
            shutil.rmtree(__nuscenes_bev_path)
        os.makedirs(__nuscenes_bev_path)

        try:
            # Load info file to access the location information
            with open(Path(f"{self.__nuscenes_info_path}.json")) as nuscenes_info_file:
                __nuscenes_info = json.load(nuscenes_info_file)
                print("Successfully opened nuScenes info file.")

                for i in tqdm(range(0, len(__scenes_list)),
                              desc='Creating BEV images for object tracker data',
                              colour='red'):

                    # Create a scene path to store the scene BEVs
                    __bev_scene_path = os.sep.join([self.__out_path, 'bev_obj_track',
                                                    __scene_names[i]])
                    if os.path.exists(__bev_scene_path):
                        shutil.rmtree(__bev_scene_path)
                    os.makedirs(__bev_scene_path)

                    for x in range(0, len(__nuscenes_info)):
                        if __nuscenes_info[x]['name'] == __scene_names[i]:
                            __scene_location = __nuscenes_info[x]['location']
                            break

                    __samples_list = sorted(os.listdir(__scenes_list[i]),
                                            key=lambda item: int(item.split('_')[0]))
                    for each_sample in __samples_list:
                        __sample_path = os.sep.join([__scenes_list[i], each_sample])
                        __data = pd.read_csv(__sample_path)
                        __path_to_save = os.sep.join([__bev_scene_path,
                                                      each_sample.split('.')[0]])
                        self.plot_bev(plotting_data=__data, location=__scene_location,
                                      bev_save_path=__path_to_save)

        except (FileNotFoundError, IsADirectoryError):
            print("NuScenes info file not found. Please provide proper directory or "
                  "run info gather step before attempting to generate BEV images.")
