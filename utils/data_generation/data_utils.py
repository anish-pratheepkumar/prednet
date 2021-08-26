"""functions for data preprocessing and post processing """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

import h5py
import numpy as np

from experiments import config


def quat2euler(q):
    """
    Converts quaternion to Euler angles in radians
    Args
      q: 1x4 quaternion
    Returns
      eul: 1x3 Euler angle representation
    """
    # arctan => taninverse
    temp1 = np.divide(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (np.square(q[1]) + np.square(q[2])))
    E1 = np.arctan(temp1)

    E2 = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))

    temp2 = np.divide(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (np.square(q[2]) + np.square(q[3])))
    E3 = np.arctan(temp2)

    eul = np.array([E1, E2, E3])
    return eul


def euler2quat(eul):
    """
    Converts Euler angles in radians to quaternion (for root orientation)
    Args
      eul: 1x3 Euler angle representation
    Returns
      q: 1x4 quaternion
    """
    (E1, E2, E3) = (eul[0], eul[1], eul[2])
    qw = np.cos(E1 / 2) * np.cos(E2 / 2) * np.cos(E3 / 2) + np.sin(E1 / 2) * np.sin(E2 / 2) * np.sin(E3 / 2)
    qx = np.sin(E1 / 2) * np.cos(E2 / 2) * np.cos(E3 / 2) - np.cos(E1 / 2) * np.sin(E2 / 2) * np.sin(E3 / 2)
    qy = np.cos(E1 / 2) * np.sin(E2 / 2) * np.cos(E3 / 2) + np.sin(E1 / 2) * np.cos(E2 / 2) * np.sin(E3 / 2)
    qz = np.cos(E1 / 2) * np.cos(E2 / 2) * np.sin(E3 / 2) - np.sin(E1 / 2) * np.sin(E2 / 2) * np.cos(E3 / 2)
    q = np.array([qw, qx, qy, qz])
    return q


def load_qpos_data(pred_data):
    """
    Args
        pred_data: 1x38 qpos data collected from mujoco
    Returns
        pred_data_radian: 1x37 un-normalized data
    """
    humanoid_qpos_data = copy.deepcopy(pred_data)  # copy list to prevent change to parent list -> save_qos_goal_pos
    # humanoid_qpos_data = pred_data
    if config.AVOID_GOAL:
        humanoid_qpos_data = humanoid_qpos_data[:, :35]  # avoid goal pos

    # Important data preprocessing: convert root orientation in quaternion to radians
    root_orien_quat = humanoid_qpos_data[:, 3:7]
    root_orien_radian = np.zeros_like(root_orien_quat)
    for i in range(humanoid_qpos_data.shape[0]):
        root_orien_radian[i][:3] = quat2euler(root_orien_quat[i])
    humanoid_qpos_data[:, 3:7] = root_orien_radian
    humanoid_qpos_data = np.delete(humanoid_qpos_data, 6, axis=1)  # 1 dimension reduced after conversion

    pred_data_radian = humanoid_qpos_data  # qpos and goal pos data. total_dim = 35-1 +3 = 37

    return pred_data_radian


def normalize_data(data, data_mean, data_std):
    """
    Normalize input data by subtracting the mean and dividing by the standard deviation

    Args
        data: qpos data
        data_mean: vector of mean used to normalize the data
        data_std: vector of standard deviation used to normalize the data
    Returns
        data_out: the passed data matrix, but normalized
    """
    if config.ARCHITECTURE == "red":
        data[:, :34] = np.divide((data[:, :34] - data_mean), data_std)

        # not normalising rootpos and goal pos
    else:
        data[:, 3:34] = np.divide((data[:, 3:34] - data_mean), data_std)
        # not normalising rootpos and goal pos
    return data


def post_process(output_data, data_mean, data_std):
    """
    post_process output data by denormalizing and converting radian root orientation back to quat

    Args
        output_data: model output having 1x34 qpos values
        data_mean: vector of mean used to denormalize the data
        data_std: vector of standard deviation used to denormalize the data
    Returns
        data_out: the passed data matrix, but denormalized
    """
    # denormalize all data qpos & goalpos
    if config.ARCHITECTURE == "red":
        output_data[:, :34] = np.multiply(output_data[:, :34], data_std) + data_mean

        # not normalising rootpos and goal pos
    else:
        output_data[:, 3:34] = np.multiply(output_data[:, 3:34], data_std) + data_mean

    # convert root orientation from radians to quaternions
    root_orien_radian = output_data[:, 3:6]
    root_orien_quat = np.zeros_like(root_orien_radian)
    root_orien_quat = np.insert(root_orien_quat, 3, 0, axis=1)  # quat has 4 dimensions
    for i in range(output_data.shape[0]):
        root_orien_quat[i] = euler2quat(root_orien_radian[i])
    final_outputs = np.insert(output_data, 6, 0, axis=1)
    final_outputs[:, 3:7] = root_orien_quat

    return final_outputs


def load_mogaze_data(path_to_dataset: str, actions, limit):
    """
    Args
      path_to_dataset: string. directory where the data resides
      actions: action to load
      limit: select the frame limit from available data
    Returns
      trainData: dictionary with key:value
        key=(action, action_sample), value=(nxd) un-normalized data
    """
    s_human_pos_gaze_data = {}
    completeData = []
    path_to_human = ''
    path_to_gaze = ''

    for action in actions:
        path_to_action = os.path.join(path_to_dataset, action)
        for action_sample in os.listdir(path_to_action):
            if 'human' in action_sample:
                sample = action_sample.split('.')[0]
                path_to_human = os.path.join(path_to_action, action_sample)
            else:
                path_to_gaze = os.path.join(path_to_action, action_sample)

        # read mogaze data from hdf5 file; h5py file like a dictionary we need its key to access the contents
        human_pos_hf = h5py.File(path_to_human, 'r')
        human_pos_key = list(human_pos_hf.keys())[0]
        human_pos_data = np.array(human_pos_hf[human_pos_key][:limit, :])

        human_gaze_hf = h5py.File(path_to_gaze, 'r')
        human_gaze_key = list(human_gaze_hf.keys())[0]
        human_gaze_data = np.array(human_gaze_hf[human_gaze_key][:limit, 2:5])

        human_pos_gaze_data = np.hstack((human_pos_data, human_gaze_data))

        if len(completeData) == 0:
            completeData = copy.deepcopy(human_pos_gaze_data)
        else:
            completeData = np.append(completeData, human_pos_gaze_data, axis=0)

        # sample the data at every 5th frame
        s_human_pos_gaze_data = human_pos_gaze_data[::5]

        s_human_pos_gaze_data = np.array_split(s_human_pos_gaze_data,
                                               30)
        s_human_pos_gaze_data = {k: v for k, v in enumerate(s_human_pos_gaze_data)}

        if config.TRAIN_MOGAZE:
            human_pos_gaze_data = np.array_split(human_pos_gaze_data,
                                                 30)  # 53250/30 = 1775 : 30 batches with 1775 timesteps
            human_pos_gaze_data = {k: v for k, v in enumerate(human_pos_gaze_data)}
            return human_pos_gaze_data, completeData

        return s_human_pos_gaze_data, completeData


def keys(f):
    """
    Get all the keys of the h5 file
    """
    return [key for key in f.keys()]


def load_hri_data(path_to_dataset: str, action="combined", dataset_type="train"):
    print("loading: ", path_to_dataset)
    h5f = h5py.File(path_to_dataset, "r")
    combined_train = h5f["{}_{}".format(action, dataset_type)]
    all_episodes = []
    # print("combined_train dataset keys: ", combined_train.keys())  # The episodes under "combined_train" dataset
    for i in range(len(keys(combined_train))):
        all_episodes.append(combined_train["episode_{}".format(i)][()])
    return all_episodes


class DataSaver:
    """saves the qpos data in specific scenario folder or subfolder"""

    def __init__(self, scenario, filename, count, filetype, data, root):
        self.root_dir = root
        self.data_folder = scenario
        self.filename = filename
        self.count = count
        self.filetype = filetype
        self.data = data

    def get_save_path(self):
        path = os.path.normpath(
            os.path.join(self.root_dir, self.data_folder))
        os.makedirs(path, exist_ok=True)
        filename_qpos = path + '/' + self.filename + str(self.count) + self.filetype
        return filename_qpos

    def save(self):
        filename_qpos = self.get_save_path()
        # try:
        #     np.savetxt(filename_qpos, self.data, delimiter=',')
        # except:
        np.savetxt(filename_qpos, np.array(self.data).squeeze(), delimiter=',')

    def add_subfolder(self, sub_folder: str):
        self.data_folder = os.path.join(self.data_folder, sub_folder)


root_dir = os.path.dirname(os.path.abspath(__file__))


def collect_qpos_goal_pos_mogaze(human_env, save_qpos_goal_pos, goal_pos=None, chull=False):
    # base_pos = human_env.base_trans_joints.reshape((1,-1))
    # other_joints_and_base_orien = human_env.non_fixed_other_joints
    joint_qpos = human_env.joint_qpos
    if chull:
        if len(save_qpos_goal_pos) == 0:
            save_qpos_goal_pos = joint_qpos.reshape(1, -1)
        else:
            save_qpos_goal_pos = np.vstack((save_qpos_goal_pos, joint_qpos))
        return save_qpos_goal_pos
    else:
        pass
        # assert goal_pos is not None
        # qpos_goal_pos = np.hstack((humanoid_qpos, goal_pos))
        # qpos_goal_pos = qpos_goal_pos.reshape(1, 38)
        # save_qpos_goal_pos = np.vstack((save_qpos_goal_pos, qpos_goal_pos))
        # return save_qpos_goal_pos


def save_qpos_chull_viz(episodes, count, pred_outputs, scenario='noise', root=None):
    # save qpos of drl_human and dl_humans for visualizing the chull vol occupancy
    pred_data_saver = DataSaver(scenario=scenario, filename='qpos_pred_chull_viz_', count=count, filetype='.csv',
                                data=pred_outputs, root=root)
    # if config.AVOID_GOAL:
    #     pred_data_saver.add_subfolder('avoid_goal')
    # pred_data_saver.add_subfolder('pred_chull_viz')
    # pred_data_saver.add_subfolder('qpos')
    pred_data_saver.add_subfolder('episode_' + str(episodes))
    pred_data_saver.save()


def collect_qpos_hri_test(human_env, save_qpos_goal_pos):
    humanoid_qpos = human_env.joint_qpos.copy()
    humanoid_qpos = humanoid_qpos.reshape(1, 35)
    if len(save_qpos_goal_pos) > 0:
        save_qpos_goal_pos = np.vstack((save_qpos_goal_pos, humanoid_qpos))
    else:
        save_qpos_goal_pos = humanoid_qpos

    return save_qpos_goal_pos
