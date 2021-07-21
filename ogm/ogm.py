# Built-in modules:
import os
import shutil
import json
from pathlib import Path

# Other dependencies:
from tqdm import tqdm
import pandas as pd
import numpy as np


class NuscenesOGM:
    def __init__(self, output_path, size_of_ogm, distance):
        self.__output_path = output_path
        self.__ogm_size = size_of_ogm
        self.__distance = distance
        self.__grid_cell_distance = tuple(int(i/j) for i, j in
                                          zip(tuple(i * 2 for i in self.__distance), self.__ogm_size))
        self.__formatted_data_path = os.sep.join([output_path, 'nuscenes_formatted_data'])

    def ogm_creator(self, input_data_frame, objects):
        # Create an empty OGM based on the desired size:
        __base_ogm = np.zeros(self.__ogm_size)

        # Calculate the center of the scene:
        __x_origin = input_data_frame['x'][0] - self.__distance[0]
        __y_origin = input_data_frame['y'][0] - self.__distance[1]

        # Create empty lists to store the grid information:
        __x_grids = []
        __y_grids = []

        for i in range(0, len(input_data_frame)):
            if input_data_frame['category'][i] in objects:
                # Grids occupied by the center of object:
                __x_grids.append((input_data_frame['x'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 1st corner (anchor-points) of the object:
                __x_grids.append((input_data_frame['x_anchor'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y_anchor'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 2nd corner of the object:
                __x_grids.append((input_data_frame['x2'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y2'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 3nd corner of the object:
                __x_grids.append((input_data_frame['x3'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y3'][i] - __y_origin) / self.__grid_cell_distance[1])

                # Grids occupied by the 3nd corner of the object:
                __x_grids.append((input_data_frame['x4'][i] - __x_origin) / self.__grid_cell_distance[0])
                __y_grids.append((input_data_frame['y4'][i] - __y_origin) / self.__grid_cell_distance[1])

        for x, y in zip(__x_grids, __y_grids):
            '''
            Note: 
            1.) The below condition filters a point if it coincides with the grid line. 
            There won't be an instance where all the points of the bounding box coincide 
            with the grid lines, so there won't be an instance where the object becomes a ghost object.
            
            2.) There might be a situation where the corner lies outside the BEV. This
            again as to be neglected. The other points will be considered and the grid
            will be marked occupied.
            '''
            if (not x.is_integer() and not y.is_integer()) and \
                    (np.ceil(x) <= self.__ogm_size[0] and np.ceil(y) <= self.__ogm_size[1]):
                __base_ogm[int(np.ceil(y)) - 1, int(np.ceil(x)) - 1] = 1

        return np.flip(__base_ogm, axis=0)

    def generate_ogm(self):
        __scene_names = []
        __scenes_list = []

        try:
            # Load list of desired objects:
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

                    # creating an output path to store BEV images
                    __nuscenes_ogm_path = os.sep.join([self.__output_path, 'ogm_nuscenes'])
                    if os.path.exists(__nuscenes_ogm_path):
                        shutil.rmtree(__nuscenes_ogm_path)
                    os.makedirs(__nuscenes_ogm_path)

                    # Looping over scenes:
                    for __each_scene, __each_scene_name in tqdm(zip(__scenes_list, __scene_names),
                                                                desc='Creating OGMs for the formatted data',
                                                                colour='red'):

                        __saving_path = os.sep.join([__nuscenes_ogm_path, __each_scene_name])
                        os.mkdir(__saving_path)

                        __samples_list = sorted(os.listdir(__each_scene),
                                                key=lambda item: int(item.split('_')[0]))

                        # Looping over samples in each list:
                        for __each_sample in __samples_list:
                            __sample_data = pd.read_csv(os.sep.join([__each_scene, __each_sample]))
                            np.save(os.sep.join([__saving_path, __each_sample.split('.')[0]]),
                                    self.ogm_creator(__sample_data, __objects))

                else:
                    print("Formatted data not found. Consider formatting the data first before"
                          " generating Occupancy Grid Maps.")

        except FileNotFoundError:
            print("ogm_objects_nuscenes.json files not found. Please check the folder.")
