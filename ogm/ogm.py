# Required built-in modules:
# -----------------------------
import os
import shutil
import json
from pathlib import Path

# Required additional libraries:
# --------------------------------
from tqdm import tqdm
import pandas as pd
import numpy as np


# Class to create OGMs:
# -------------------------
class NuscenesOGM:
    """
    Code to generate OGMs which are used as targets during training and testing.
    The same OGMs also serve as ground-truth for object-tracker experiments.
    """

    def __init__(self, output_path, size_of_ogm, distance):
        """
        :param output_path: Desired path where the generated OGMs will be stored

        :param size_of_ogm: The height * width of desired OGMs. Height and width
        need not be same

        :param distance: The distance covered by the OGM from the center.
        Should be same as the distance covered by the corresponding BEVs
        """
        self.__output_path = output_path
        self.__ogm_size = size_of_ogm
        self.__distance = distance
        self.__grid_cell_distance = tuple(int(i/j) for i, j in
                                          zip(tuple(i * 2 for i in self.__distance), self.__ogm_size))
        self.__formatted_data_path = os.sep.join([output_path, 'nuscenes_formatted_data'])


    def ogm_creator(self, input_data_frame, objects):
        """
        The function is used to assign occupancy to the grid cells of OGMs

        :param input_data_frame: A Pandas dataframe for a particular nuScenes sample

        :param objects: Objects of interest that needs to be considered during assignment

        :return: An OGM with assigned occupancies
        """

        # Create an empty OGM based on the desired size:
        # ------------------------------------------------------
        __base_ogm = np.zeros(self.__ogm_size)

        # Calculate the center of the scene:
        # --------------------------------------
        __x_origin = input_data_frame['x'][0] - self.__distance[0]
        __y_origin = input_data_frame['y'][0] - self.__distance[1]

        # Create empty lists to store the grid information:
        # -------------------------------------------------------
        __x_grids = []
        __y_grids = []

        for i in range(0, len(input_data_frame)):
            if input_data_frame['category'][i] in objects:
                # Grids occupied by the center of object:
                # ---------------------------------------------
                __x_grids.append((input_data_frame['x'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 1st corner (anchor-points) of the object:
                # -----------------------------------------------------------------------
                __x_grids.append((input_data_frame['x_anchor'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y_anchor'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 2nd corner of the object:
                # ------------------------------------------------------
                __x_grids.append((input_data_frame['x2'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y2'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 3rd corner of the object:
                # -----------------------------------------------------
                __x_grids.append((input_data_frame['x3'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y3'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 4th corner of the object:
                # ------------------------------------------------------
                __x_grids.append((input_data_frame['x4'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y4'][i] - __y_origin) / self.__grid_cell_distance[1])

        for x, y in zip(__x_grids, __y_grids):
            '''
            NOTE: 
            1.) The below condition filters a point if it coincides with the grid line. 
            There won't be an instance where all the points of the bounding box coincide 
            with the grid lines, so there won't be an instance where the object becomes a ghost object.
            
            2.) There might be a situation where the corner lies outside the BEV. This
            again has to be neglected. The other points will be considered and the grid
            will be marked occupied.
            '''
            if x > 0 and y > 0:
                if (not x.is_integer() and not y.is_integer()) and (np.ceil(x) <= self.__ogm_size[0]
                                                                    and np.ceil(y) <= self.__ogm_size[1]):
                    __base_ogm[int(np.ceil(y)) - 1, int(np.ceil(x)) - 1] = 1

        return np.flip(__base_ogm, axis=0)


    def generate_ogm(self):
        """
        The main function used to generate OGMs

        :return: Generates OGMs and saves the OGMs as numpy arrays in a defined output directory
        """
        __scene_names = []
        __scenes_list = []

        try:
            # Load list of desired objects:
            # --------------------------------
            __objects_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'ogm_objects_nuscenes.json')
            with open(Path(__objects_path)) as __objects_file:
                __objects = json.load(__objects_file)

                if Path(self.__formatted_data_path).exists():
                    print("Formatted data for nuscenes found.")
                    for path, subdir, files in os.walk(self.__formatted_data_path):
                        for name in sorted(subdir):
                            __scene_names.append(name)
                            __scenes = os.path.join(path, name)
                            __scenes_list.append(__scenes)

                    # Creating an output path to store OGMs:
                    # ---------------------------------------------
                    __nuscenes_ogm_path = os.sep.join([self.__output_path, 'ogm_nuscenes'])
                    if os.path.exists(__nuscenes_ogm_path):
                        shutil.rmtree(__nuscenes_ogm_path)
                    os.makedirs(__nuscenes_ogm_path)

                    # Looping over scenes:
                    # -----------------------
                    for __each_scene, __each_scene_name in zip(tqdm(__scenes_list,
                                                                    desc='Creating OGMs for nuScenes data',
                                                                    colour='blue'), __scene_names):

                        __saving_path = os.sep.join([__nuscenes_ogm_path, __each_scene_name])
                        os.mkdir(__saving_path)

                        __samples_list = sorted(os.listdir(__each_scene),
                                                key=lambda item: int(item.split('_')[0]))

                        # Looping over samples in each list:
                        # ------------------------------------
                        for __each_sample in __samples_list:
                            __sample_data = pd.read_csv(os.sep.join([__each_scene, __each_sample]))
                            np.save(os.sep.join([__saving_path, __each_sample.split('.')[0]]),
                                    self.ogm_creator(__sample_data, __objects))

                else:
                    print("Formatted data not found. Consider formatting the data first before"
                          " generating Occupancy Grid Maps.")

        except FileNotFoundError:
            print("ogm_objects_nuscenes.json files not found. Please check the folder.")
