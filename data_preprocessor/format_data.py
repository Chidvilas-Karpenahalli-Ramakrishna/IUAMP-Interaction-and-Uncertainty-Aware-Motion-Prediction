# Required built-in modules:
# -----------------------------
import json
from pathlib import Path
import os
import shutil
import gc

# Additional required libraries and modules:
# ----------------------------------------------
import pandas as pd
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes


# Data formatting class for nuScenes data:
# ----------------------------------------------
class NuScenesDataFormat:
    """
    The Class is meant to format the data to a tabular format where each sample
    info is stored as a .csv file under each scene directory. This is done for ease of 
    access and ease of generating BEV images.
    """

    def __init__(self, input_path, output_path, version, gathered_info_path):
        """
        :param input_path: The path to the directory containing nuScenes meta-data

        :param output_path: The path to the directory where the output files will be stored.

        :param version: The version of the nuScenes dataset that is downloaded

        :param gathered_info_path: The path to the directory where info files are stored.
        """
        self.__gathered_info_path = gathered_info_path
        self.__meta_data = input_path
        self.__output_path = output_path
        self.__data_version = version
        self.__json_files = []
        self.__info_file = ''


    def read_files(self):
        """
        This method makes sure that the information file generated from the
        info_generation step is present in the directory and the corresponding
        meta-data is present in the input-directory.
        """
        try:
            self.__json_files += [each for each in os.listdir(self.__gathered_info_path)
                                  if each.endswith('.json')]
            print(f"The following info files were found in the directory and data formatting "
                  f"will be attempted for them: {self.__json_files}")
        except FileNotFoundError:
            print("The info file was not found in the directory. Consider gathering "
                  "information before data formatting or check the path.")

        for each_file in self.__json_files:
            __info_file_version = each_file.split('.json')[0]
            try:
                if __info_file_version == self.__data_version:
                    print(f"The corresponding meta data for {__info_file_version} dataset was found.")
                    self.__info_file = each_file
            except FileNotFoundError:
                print(f"The corresponding meta data for {__info_file_version} dataset was not found.")


    @staticmethod
    def get_euler_angles(quaternion):
        """
        Takes a 4D quaternion and returns only the yaw angle in radians. Just add
        additional code to get pitch and roll if needed.

        :param quarternion: The quarternion info given as a list in nuScenes dataset
        """
        __numerator = 2 * (quaternion[0]*quaternion[3]+quaternion[1]*quaternion[2])
        __denominator = 1 - (2*(np.square(quaternion[2])+np.square(quaternion[3])))
        __yaw_angle = np.arctan2(__numerator, __denominator)
        return __yaw_angle


    @staticmethod
    def bounding_box_info(input_data):
        """
        The method accesses the annotation information and calculates the heading
        angle and anchor points needed to create bounding boxes in BEV images.

        :param input_data: A pandas dataframe
        """
        __bounding_box = {'x_anchor': [], 'y_anchor': [], 'heading_angle': [],
                          'x2': [], 'y2': [], 'x3': [], 'y3': [], 'x4': [], 'y4': []}
        for i in range(0, len(input_data['rotation'])):
            __heading_angle = NuScenesDataFormat.get_euler_angles(
                input_data['rotation'][i])

            # Here x-anchor and y-anchor are needed to draw rectangles:
            __x_anchor = input_data['x'][i] - (((input_data['width'][i] / 2) *
                                                np.cos(__heading_angle)) -
                                               ((input_data['height'][i] / 2) *
                                                np.sin(__heading_angle)))
            __y_anchor = input_data['y'][i] - (((input_data['width'][i] / 2) *
                                                np.sin(__heading_angle)) +
                                               ((input_data['height'][i] / 2) *
                                                np.cos(__heading_angle)))
            __bounding_box['x_anchor'].append(__x_anchor)
            __bounding_box['y_anchor'].append(__y_anchor)
            __bounding_box['heading_angle'].append(np.degrees(__heading_angle))

            # Find other co-ordinates of rectangle which will be used later to create OGMs:
            __x2 = input_data['x'][i] + (((input_data['width'][i] / 2) *
                                          np.cos(__heading_angle)) +
                                         ((input_data['height'][i] / 2) *
                                          np.sin(__heading_angle)))
            __y2 = input_data['y'][i] + (((input_data['width'][i] / 2) *
                                          np.sin(__heading_angle)) -
                                         ((input_data['height'][i] / 2) *
                                         np.cos(__heading_angle)))
            __bounding_box['x2'].append(__x2)
            __bounding_box['y2'].append(__y2)

            __x3 = input_data['x'][i] + (((input_data['width'][i] / 2) *
                                          np.cos(__heading_angle)) -
                                         ((input_data['height'][i] / 2) *
                                          np.sin(__heading_angle)))
            __y3 = input_data['y'][i] + (((input_data['width'][i] / 2) *
                                          np.sin(__heading_angle)) +
                                         ((input_data['height'][i] / 2) *
                                         np.cos(__heading_angle)))
            __bounding_box['x3'].append(__x3)
            __bounding_box['y3'].append(__y3)

            __x4 = input_data['x'][i] - (((input_data['width'][i] / 2) *
                                          np.cos(__heading_angle)) +
                                         ((input_data['height'][i] / 2) *
                                          np.sin(__heading_angle)))
            __y4 = input_data['y'][i] - (((input_data['width'][i] / 2) *
                                          np.sin(__heading_angle)) -
                                         ((input_data['height'][i] / 2) *
                                         np.cos(__heading_angle)))
            __bounding_box['x4'].append(__x4)
            __bounding_box['y4'].append(__y4)

        __bounding_box_data = pd.DataFrame(__bounding_box)
        __data = pd.concat([input_data, __bounding_box_data], axis=1, join='inner')

        return __data


    def data_formatter(self):
        """
        The method is the main method in the Class. It accesses all the files needed
        for data formatting and calls other static methods to perform the task.
        """
        print("Started data formatting.")
        try:
            nusc = NuScenes(version=self.__data_version, dataroot=self.__meta_data, verbose=True)
            with open(Path(os.sep.join([self.__gathered_info_path, self.__info_file]))) as file, \
                    open(Path(os.sep.join([self.__meta_data, self.__data_version, 'ego_pose.json']))) as ego:
                info = json.load(file)
                ego_info = json.load(ego)
                try:
                    out_path = os.sep.join([self.__output_path, "nuscenes_formatted_data"])
                    if os.path.exists(out_path):
                        shutil.rmtree(out_path)
                    os.makedirs(out_path)
                except OSError:
                    print("Could not create an output directory to store formatted data.")
                else:
                    print("Successfully created output directory to store formatted data.")

                for each_scene in tqdm(info, desc='Formatting data', colour='cyan'):
                    __scene_name = each_scene['name']
                    __scene_path = os.sep.join([out_path, __scene_name])

                    if os.path.exists(__scene_path):
                        shutil.rmtree(__scene_path)
                    os.makedirs(__scene_path)
                    __samples = each_scene['sample_tokens']
                    __ego_tokens = each_scene['ego_pose_tokens']

                    __count = 0
                    for i in range(0, len(__samples)):
                        __count += 1
                        __sample_annotations = nusc.get('sample', __samples[i])['anns']
                        __annotation_data = {'x': [], 'y': [], 'rotation': [], 'height': [],
                                             'width': [], 'category': [], 'visibility': []}

                        for each_token in ego_info:
                            if each_token['token'] == __ego_tokens[i]:
                                __annotation_data['x'].append(each_token['translation'][0])
                                __annotation_data['y'].append(each_token['translation'][1])
                                __annotation_data['rotation'].append(
                                    np.array(each_token['rotation'], dtype=float))
                                # dimension as per EGO model (Renault zoe)
                                __annotation_data['height'].append(1.730)
                                __annotation_data['width'].append(4.084)

                                __annotation_data['category'].append('ego')
                                __annotation_data['visibility'].append(4)
                                break

                        for each_annotation in __sample_annotations:
                            __meta_data = nusc.get('sample_annotation', each_annotation)
                            __annotation_data['x'].append(__meta_data['translation'][0])
                            __annotation_data['y'].append(__meta_data['translation'][1])
                            __annotation_data['rotation'].append(
                                np.array(__meta_data['rotation'], dtype=float))
                            __annotation_data['height'].append(__meta_data['size'][0])
                            __annotation_data['width'].append(__meta_data['size'][1])
                            __annotation_data['category'].append(__meta_data['category_name'])
                            __annotation_data['visibility'].append(__meta_data['visibility_token'])

                        __raw_data = pd.DataFrame(__annotation_data, dtype=float)
                        __data = NuScenesDataFormat.bounding_box_info(input_data=__raw_data)

                        __data.to_csv(
                            path_or_buf=os.sep.join([__scene_path, f"{__count}_{__samples[i]}.csv"]),
                            index=False, encoding='utf-8')

            """
            Free some memory. Unreachable objects from the NuScenes module can 
            cause memory issues and trigger:
            exit code 137 (interrupted by signal 9: SIGKILL) on Pycharm
            """
            del nusc, info, ego_info, file, ego, __annotation_data, __raw_data, __data
            gc.collect()

        except IsADirectoryError:
            print("The gathered info file was not found. Please consider gathering "
                  "information before data formatting.")



# Object tracker formatting class:
# -------------------------------------
class ObjTrackDataFormat:
    """
    The Class is meant to format the data to a tabular format where each sample
    info is stored as a .csv file under each scene directory for object tracker data. 
    This is done for ease of access and ease of generating object trcaker BEV images.
    """

    def __init__(self, input_path, output_path, version, gathered_info_path):
        """
        :param input_path: The path to the directory containing nuScenes meta-data

        :param output_path: The path to the directory where the output files will be stored.

        :param version: The version of the nuScenes dataset that is downloaded

        :param gathered_info_path: The path to the directory where info files are stored.
        """
        self.__gathered_info_path = gathered_info_path
        self.__meta_data = input_path
        self.__output_path = output_path
        self.__data_version = version
        self.__info_files = []
        self.__info_file = ''


    def read_files(self):
        """
        The method makes sure the file from info_gather step is present in the
        directory.
        """
        try:
            self.__info_files += [each for each in os.listdir(self.__gathered_info_path)
                                  if each.endswith('.json')]
            print(f"The following info files were found in the directory: "
                  f"{self.__info_files}")
        except FileNotFoundError:
            print("No info files were found in the directory. Consider gathering "
                  "information before data formatting or check the path.")

        for each_file in self.__info_files:
            try:
                if each_file == 'obj_track_info.json':
                    print(f"The object tracker info file '{each_file}' was found. "
                          f"Proceeding to data formatting.")
                    self.__info_file = each_file
            except FileNotFoundError:
                print("The object tracker info file was not found. Data formatting will "
                      "be halted. Please perform info gathering before attempting data "
                      "formatting or check the path provided for the info file.")


    @staticmethod
    def get_euler_angles(quaternion):
        """
        Takes a 4D quaternion and returns only the yaw angle in radians. Just add
        additional code to get pitch and roll if needed.

        :param quarternion: A list containing the quarternion info provided by the object tracker.
        """
        __numerator = 2 * (quaternion[0]*quaternion[3]+quaternion[1]*quaternion[2])
        __denominator = 1 - (2*(np.square(quaternion[2])+np.square(quaternion[3])))
        __yaw_angle = np.arctan2(__numerator, __denominator)
        return __yaw_angle


    @staticmethod
    def bounding_box_info(input_data):
        """
        The method accesses the annotation information and calculates the heading
        angle and anchor points needed to create BEV images.

        :param input_data: Apandas dataframe.
        """
        __bounding_box = {'x_anchor': [], 'y_anchor': [], 'heading_angle': [],
                          'x2': [], 'y2': [], 'x3': [], 'y3': [], 'x4': [], 'y4': []}
        for i in range(0, len(input_data['rotation'])):
            __heading_angle = ObjTrackDataFormat.get_euler_angles(
                input_data['rotation'][i])
            __x_anchor = input_data['x'][i] - (((input_data['width'][i] / 2) *
                                                np.cos(__heading_angle * (np.pi / 180))) -
                                               ((input_data['height'][i] / 2) *
                                               np.sin(__heading_angle * (np.pi / 180))))
            __y_anchor = input_data['y'][i] - (((input_data['width'][i] / 2) *
                                               np.sin(__heading_angle * (np.pi / 180))) +
                                               ((input_data['height'][i] / 2) *
                                               np.cos(__heading_angle * (np.pi / 180))))

            __bounding_box['x_anchor'].append(__x_anchor)
            __bounding_box['y_anchor'].append(__y_anchor)
            __bounding_box['heading_angle'].append(np.degrees(__heading_angle))

            # Find other co-ordinates of rectangle to create OGMs:
            __x2 = input_data['x'][i] + (((input_data['width'][i] / 2) *
                                          np.cos(__heading_angle)) +
                                         ((input_data['height'][i] / 2) *
                                          np.sin(__heading_angle)))
            __y2 = input_data['y'][i] + (((input_data['width'][i] / 2) *
                                          np.sin(__heading_angle)) -
                                         ((input_data['height'][i] / 2) *
                                          np.cos(__heading_angle)))
            __bounding_box['x2'].append(__x2)
            __bounding_box['y2'].append(__y2)

            __x3 = input_data['x'][i] + (((input_data['width'][i] / 2) *
                                          np.cos(__heading_angle)) -
                                         ((input_data['height'][i] / 2) *
                                          np.sin(__heading_angle)))
            __y3 = input_data['y'][i] + (((input_data['width'][i] / 2) *
                                          np.sin(__heading_angle)) +
                                         ((input_data['height'][i] / 2) *
                                          np.cos(__heading_angle)))
            __bounding_box['x3'].append(__x3)
            __bounding_box['y3'].append(__y3)

            __x4 = input_data['x'][i] - (((input_data['width'][i] / 2) *
                                          np.cos(__heading_angle)) +
                                         ((input_data['height'][i] / 2) *
                                          np.sin(__heading_angle)))
            __y4 = input_data['y'][i] - (((input_data['width'][i] / 2) *
                                          np.sin(__heading_angle)) -
                                         ((input_data['height'][i] / 2) *
                                          np.cos(__heading_angle)))
            __bounding_box['x4'].append(__x4)
            __bounding_box['y4'].append(__y4)

        __bounding_box_data = pd.DataFrame(__bounding_box)
        __data = pd.concat([input_data, __bounding_box_data], axis=1, join='inner')

        return __data


    def data_formatter(self):
        """
        This is the main method of the Class which calls other staticmethods creates data-
        formatted files for object tracker data.
        """
        print("Started data formatting.")
        __obj_track_info_path = os.sep.join([self.__output_path, self.__info_file])
        __ego_pose_path = os.sep.join(
            [self.__meta_data, self.__data_version, 'ego_pose.json'])

        try:
            with open(Path(__obj_track_info_path)) as obj_track_file, \
                    open(Path(__ego_pose_path)) as ego_pose_file:
                __obj_track_info = json.load(obj_track_file)
                __ego_pose_info = json.load(ego_pose_file)

                try:
                    out_path = os.sep.join(
                        [self.__output_path, 'obj_track_formatted_data'])
                    if os.path.exists(out_path):
                        shutil.rmtree(out_path)
                    os.makedirs(out_path)
                except OSError:
                    print("Could not create an output directory to store object tracker "
                          "formatted data.")
                else:
                    print("Successfully created output directory to store object tracker "
                          "formatted data.")

                for each_scene in tqdm(__obj_track_info,
                                       desc='Formatting object tracker data', colour='blue'):
                    __scene_name = list(each_scene.keys())[0]
                    __scene_path = os.sep.join([out_path, __scene_name])

                    if len(each_scene[list(each_scene.keys())[0]]) != 0:
                        if os.path.exists(__scene_path):
                            shutil.rmtree(__scene_path)
                        os.makedirs(__scene_path)

                        __samples_info = each_scene[list(each_scene.keys())[0]]

                        count = 0
                        for each_sample in __samples_info:
                            count += 1
                            __tracking_data = {'x': [], 'y': [], 'rotation': [], 'height': [],
                                               'width': [], 'category': [], 'tracking_score': []}
                            __sample_name = each_sample[1][1]['sample_token']
                            __traffic_agents_info = each_sample[1]
                            __ego_pose_token = each_sample[0]['ego_pose_token']

                            for each_token in __ego_pose_info:
                                if each_token['token'] == __ego_pose_token:
                                    __tracking_data['x'].append(each_token['translation'][0])
                                    __tracking_data['y'].append(each_token['translation'][1])
                                    __tracking_data['rotation'].append(
                                        np.array(each_token['rotation'], dtype=float))
                                    
                                    # Dimension are defined as per nuScenes EGO car (Renault zoe)
                                    __tracking_data['height'].append(1.730)
                                    __tracking_data['width'].append(4.084)
                                    __tracking_data['category'].append('ego')
                                    __tracking_data['tracking_score'].append(1.0)
                                    break

                            for each_agent in __traffic_agents_info:
                                """
                                Threshold is set to 30% confidence. Only those tracked 
                                traffic agents that have a confidence of more than 30% are
                                taken into consideration.
                                """
                                if each_agent['tracking_score'] >= 0.3:
                                    __tracking_data['x'].append(each_agent['translation'][0])
                                    __tracking_data['y'].append(each_agent['translation'][1])
                                    __tracking_data['rotation'].append(
                                        np.array(each_agent['rotation'], dtype=float))
                                    __tracking_data['height'].append(each_agent['size'][0])
                                    __tracking_data['width'].append(each_agent['size'][1])
                                    __tracking_data['category'].append(
                                        each_agent['tracking_name'])
                                    __tracking_data['tracking_score'].append(
                                            each_agent['tracking_score'])

                            __tracked_data = pd.DataFrame(__tracking_data)
                            __formatted_data = self.bounding_box_info(input_data=__tracked_data)

                            __file_name = f"{count}_{__sample_name}.csv"
                            __saving_path = os.sep.join([__scene_path, __file_name])
                            __formatted_data.to_csv(path_or_buf=__saving_path,
                                                    index=False, encoding='utf-8')
            
            """
            Free some memory. Unreachable objects from the NuScenes module can 
            cause memory issues and trigger:
            exit code 137 (interrupted by signal 9: SIGKILL) on Pycharm
            """
            del __obj_track_info, __ego_pose_info, __formatted_data, __tracked_data, \
                __tracking_data, __samples_info
            gc.collect()

        except IsADirectoryError:
            print("The gathered info file was not found. Please consider gathering "
                  "information before data formatting.")
