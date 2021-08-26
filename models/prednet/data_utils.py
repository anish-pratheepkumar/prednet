"""functions for data preprocessing and post processing """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

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


def load_real_data(path_to_dataset, action, category='train'):
    """
    Args
      path_to_dataset: string. directory where the data is stored
      action: action/scenario to load
      category: whether to load test data or train data
    Returns
      train_data: dictionary with key:value
        key=(action, action_sample), value=(nx35) un-normalized data
    """
    real_data = {}

    # path_to_action = os.path.join(path_to_dataset, action, category)
    h5f = h5py.File(path_to_dataset, "r")
    all_episodes = h5f["{}_{}".format(action, category)]

    for sample, humanoid_qpos_data_h5 in enumerate(all_episodes.values()):
        humanoid_qpos_data = humanoid_qpos_data_h5[()]
        humanoid_qpos_data = humanoid_qpos_data[:, :35]
        real_data[(action, sample)] = humanoid_qpos_data
    return real_data


def load_data(path_to_dataset, action, category='train'):
    """
    Args
      path_to_dataset: string. directory where the data is stored
      action: action/scenario to load
      category: whether to load test data or train data
    Returns
      trainData: dictionary with key:value
        key=(action, action_sample), value=(nx37) un-normalized data
    """
    train_data = {}
    complete_data = []
    # path_to_action = os.path.join(path_to_dataset, action, category)
    h5f = h5py.File(path_to_dataset, "r")
    all_episodes = h5f["{}_{}".format(action, category)]
    # for action_sample in natsorted(os.listdir(path_to_action)):
    # for action in ["co-existence", "co-operation", "noise"]:
    for sample, kinematic_state_data_h5 in enumerate(all_episodes.values()):
        # sample = action_sample.split('.')[0]
        # path_to_sample = os.path.join(path_to_action, action_sample)
        # kinematic_state_data = np.genfromtxt(path_to_sample, delimiter=',')
        kinematic_state_data = kinematic_state_data_h5[()]
        if config.AVOID_GOAL:
            kinematic_state_data = kinematic_state_data[:, :35]  # avoid goal pos
        else:
            kinematic_state_data = kinematic_state_data[:, :38]
        # convert root orientation in quaternion to radians
        root_orien_quat = kinematic_state_data[:, 3:7]
        root_orien_radian = np.zeros_like(root_orien_quat)
        for i in range(kinematic_state_data.shape[0]):
            root_orien_radian[i][:3] = quat2euler(root_orien_quat[i])
        kinematic_state_data[:, 3:7] = root_orien_radian
        kinematic_state_data = np.delete(kinematic_state_data, 6, axis=1)
        train_data[(action, sample)] = kinematic_state_data

        if len(complete_data) == 0:
            complete_data = copy.deepcopy(kinematic_state_data)
        else:
            complete_data = np.append(complete_data, kinematic_state_data, axis=0)

    return train_data, complete_data


def normalization_stats(complete_data):
    """"
    Args
      complete_data: nx31 matrix with data to normalize (root and goal pos not normalized)
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
    """
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)

    return data_mean, data_std


def normalize_data(data, data_mean, data_std):
    """
    Args
      data: dict with each key having 34 qpos values for HRI scenario data and 66 values for Mogaze data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_len = config.MOGAZE_SIZE if config.TRAIN_MOGAZE or config.TEST_MOGAZE else config.HUMAN_SIZE
    if isinstance(data, dict):
        for key in data.keys():
            data[key][:, 3:data_len] = np.divide((data[key][:, 3:data_len] - data_mean), data_std)
    else:
        data[..., 3:data_len] = np.divide((data[..., 3:data_len] - data_mean), data_std)

    return data


def post_process(output_data, data_mean, data_std):
    """
    Args
      output_data: model output having 1x34 qpos values
      data_mean: vector of mean used to denormalize the data
      data_std: vector of standard deviation used to denormalize the data
    Returns
      data_out: the passed data matrix, but denormalized and root orientation in quat
    """
    for i in range(len(output_data)):
        # denormalize data qpos
        output_data[i][:, 3:34] = np.multiply(output_data[i][:, 3:34], data_std) + data_mean

        # convert root orientation from radians to quaternions
        root_orien_radian = output_data[i][:, 3:6]
        root_orien_quat = np.zeros_like(root_orien_radian)
        root_orien_quat = np.insert(root_orien_quat, 3, 0, axis=1)
        for j in range(output_data[i].shape[0]):
            root_orien_quat[j] = euler2quat(root_orien_radian[j])
        output_data[i] = np.insert(output_data[i], 6, 0, axis=1)
        output_data[i][:, 3:7] = root_orien_quat
    return output_data


def post_process_mogaze(output_data, data_mean, data_std):
    """
  post_process output data by denormalizing and converting radian root orientation back to quat

  Args
      output_data: model output having 1x34 qpos values
      data_mean: vector of mean used to denormalize the data
      data_std: vector of standard deviation used to denormalize the data
  Returns
      data_out: the passed data matrix, but denormalized
  """
    for i in range(len(output_data)):
        # denormalize all data qpos & goalpos
        output_data[i][:, 3:66] = np.multiply(output_data[i][:, 3:66], data_std) + data_mean

    return output_data
