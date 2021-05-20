# In-built modules and packages:
import json
from pathlib import Path
import os

# Additional modules and packages:
from tqdm import tqdm


class NuScenesInfo:
    """
    Class gathers the required tokens and structures them scene-wise and in turn
    sample-wise for easy access during the data formatting and BEV generation step
    """
    def __init__(self, input_path):
        self.input_path = input_path
        self.scene_info = []
        self.scene_samples = []
        self.locations = []
        self.ego_pose_tokens = []

    def scene_list_getter(self):
        """
        Method gets the list of scenes in the downloaded version of nuScenes data.
        """
        try:
            with open(Path(os.sep.join([str(self.input_path), 'scene.json']))) as scene_file:
                scene_info = json.load(scene_file)
                for each_scene in tqdm(scene_info, desc='Gathering scene info',
                                       colour='green'):
                    self.scene_info.append({'name': each_scene['name'],
                                            'scene_token': each_scene['token'],
                                            'log_token': each_scene['log_token']})
            return self.scene_info

        except FileNotFoundError:
            print("scene.json file not found in the directory.")

    def sample_token_getter(self, scene_info):
        """
        Each scene has approximately 40 samples that are annotated. The present
        method gets all the sample tokens for a particular scene.
        """
        try:
            with open(Path(os.sep.join([str(self.input_path), 'sample.json']))) as sample_file:
                sample_info = json.load(sample_file)
        except FileNotFoundError:
            print("sample.json file not found in the directory.")

        for each_scene in tqdm(scene_info, desc='Gathering sample list',
                               colour='blue'):
            sample_token_list = []
            scene_token = each_scene['scene_token']
            for each_sample in sample_info:
                if each_sample['scene_token'] == scene_token:
                    sample_token_list.append(each_sample['token'])
            self.scene_samples.append({'sample_tokens': sample_token_list})

        for a, b in zip(self.scene_info, self.scene_samples):
            a.update(b)
        del self.scene_samples
        return self.scene_info

    def location_getter(self, scene_info):
        """
        To generate BEV images with an underlying map, the location in which the
        scene was recorded needs to be know. The present method gathers the
        location information for each scene.
        """
        try:
            with open(Path(os.sep.join([str(self.input_path), 'log.json']))) as log_file:
                location_info = json.load(log_file)
        except FileNotFoundError:
            print("log.json file not found in the directory.")

        for each_scene in tqdm(scene_info, desc='Gathering location info'):
            log_token = each_scene['log_token']
            for each_location in location_info:
                if each_location['token'] == log_token:
                    self.locations.append({'location': each_location['location']})
                    break

        for a, b in zip(self.scene_info, self.locations):
            a.update(b)
        del self.locations
        return self.scene_info

    def ego_pose_token_getter(self, scene_info):
        """
        The BEV images are plotted along with the EGO vehicle at the centre. The
        present method accesses the EGO token with which the EGO location and
        orientation can be obtained.
        """
        try:
            with open(Path(os.sep.join([str(self.input_path), 'sample_data.json']))) as sample_data_file:
                sample_data_info = json.load(sample_data_file)
        except FileNotFoundError:
            print("sample_data.json file not found in the directory.")

        for each_scene in tqdm(scene_info, desc='Gathering ego-pose info', colour='yellow'):
            samples_list = each_scene['sample_tokens']
            ego_pose_list = []
            for each_sample in samples_list:
                for samples in sample_data_info:
                    if samples['sample_token'] == each_sample:
                        ego_pose_list.append(samples['ego_pose_token'])
                        break

            self.ego_pose_tokens.append({'ego_pose_tokens': ego_pose_list})

        for a, b in zip(self.scene_info, self.ego_pose_tokens):
            a.update(b)
        del self.ego_pose_tokens
        return self.scene_info


class ObjTrackInfo:
    """
    The Class has methods that are responsible for rearranging the output of the
    object tracker with the nuScenes information for ease of access. The object
    tracker output is restructured into a .json file where the tracked information
    is stored scene-wise.
    """
    def __init__(self, nuscenes_info_path, obj_track_input_path, data_version):
        self.__nuscenes_info_path = nuscenes_info_path
        self.__obj_track_input_path = obj_track_input_path
        self.__data_version = data_version
        self.__tracked_data = []

    def get_scene_sample_info(self):
        """
        The method groups the output of the object tracker into respective scenes
        and the tracked samples within the scenes.
        """
        print("Started gathering information for object-tracker.")
        try:
            __nuscenes_info_path = os.sep.join([self.__nuscenes_info_path,
                                                f'{self.__data_version}.json'])
            with open(Path(self.__obj_track_input_path)) as obj_track_file, \
                    open(Path(__nuscenes_info_path)) as nuscenes_file:
                __obj_track_info = json.load(obj_track_file)['results']
                __nuscenes_info = json.load(nuscenes_file)

                for each_scene in tqdm(__nuscenes_info,
                                       desc='Scanning tracked scenes', colour='magenta'):
                    __scene_info = {each_scene['name']: []}
                    __scene_name = each_scene['name']
                    __samples_list = each_scene['sample_tokens']
                    __ego_pose_tokens = each_scene['ego_pose_tokens']

                    for i in range(0, len(__samples_list)):
                        if __samples_list[i] in __obj_track_info:
                            __data_to_append = [{'ego_pose_token': __ego_pose_tokens[i]},
                                                __obj_track_info[__samples_list[i]]]
                            __scene_info[each_scene['name']]. \
                                append(__data_to_append)

                    self.__tracked_data.append(__scene_info)

                return self.__tracked_data

        except (FileNotFoundError, IsADirectoryError):
            print("Files not found in the directory. Please check the path of "
                  "object tracker output .json file and the nuscenes_info file .json path")
