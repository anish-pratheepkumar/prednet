"""
taken from red (https://github.com/una-dinosauria/human-motion-prediction)
added codes for processing HRI scenario data and Mogaze data
Simple code for training an RNN for motion prediction.

Functions that help with data processing for human3.6m
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import h5py
import numpy as np
from natsort import natsorted
from six.moves import xrange  # pylint: disable=redefined-builtin
import copy

from experiments import config


def rotmat2euler(R):
    """
  Converts a rotation matrix to Euler angles
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args
    R: a 3x3 rotation matrix
  Returns
    eul: a 3x1 Euler angle representation of R
  """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2]);

        if R[0, 2] == -1:
            E2 = np.pi / 2;
            E1 = E3 + dlta;
        else:
            E2 = -np.pi / 2;
            E1 = -E3 + dlta;

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3]);
    return eul


def quat2expmap(q):
    """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])  # magnitude of vector part of quat - qv
    coshalftheta = q[0]  # the real part - qw

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def rotmat2quat(R):
    """
  Converts a rotation matrix to a quaternion
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

  Args
    R: 3x3 rotation matrix
  Returns
    q: 1x4 quaternion
  """
    rotdiff = R - R.T;

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2;
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps);

    costheta = (np.trace(R) - 1) / 2;

    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R));


def expmap2rotmat(r):
    """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def quat2euler(q):
    """
  Converts quaternion to Euler angles in radians
  This function gives result between -pi/2 and pi/2 only
  Applied here to convert quaternion root orientation in qpos to euler angles
  ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
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
Converts Euler angles in radians to quaternion
Applied here to convert euler angles to quaternion root orientation
ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
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


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
  Converts the output of the neural network to a format that is more easy to
  manipulate for, e.g. conversion to other format or visualization

  Args
    poses: The output from the TF model. A list with (seq_length) entries,
    each with a (batch_size, dim) output
  Returns
    poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
    batch is an n-by-d sequence of poses.
  """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in xrange(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def readCSVasFloat(filename):
    """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def load_real_data(path_to_dataset, actions, one_hot, category='train'):
    """
    Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Args
      path_to_dataset: string. directory where the data resides
      actions: list of string. The actions to load
      one_hot: Whether to add a one-hot encoding to the data
      category: variable to indicate whether to load test data or train data
    Returns
      trainData: dictionary with key:value
        key=(action, action_sample), value=(nxd) un-normalized data
    """
    # nactions = len(actions)
    real_data = {}
    completeData = []

    for action in actions:
        path_to_action = os.path.join(path_to_dataset, action, category)
        print(path_to_action)
        for action_sample in natsorted(os.listdir(path_to_action)):
            sample = action_sample.split('.')[0]
            path_to_sample = os.path.join(path_to_action, action_sample)
            humanoid_qpos_data = np.genfromtxt(path_to_sample, delimiter=',')

            # # remove qvel from wiping data
            # humanoid_qpos_data = np.delete(humanoid_qpos_data, np.s_[35:69], axis=1)

            # remove cube pos and goal pos from co-existing data
            humanoid_qpos_data = humanoid_qpos_data[:, :35]
            real_data[(action, sample)] = humanoid_qpos_data

    return real_data


def load_qpos_data(path_to_dataset, actions, one_hot, category='train'):
    """
    Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

    Args
      path_to_dataset: string. directory where the data resides
      actions: list of string. The actions to load
      one_hot: Whether to add a one-hot encoding to the data
      category: variable to indicate whether to load test data or train data
    Returns
      trainData: dictionary with key:value
        key=(action, action_sample), value=(nxd) un-normalized data
    """
    # nactions = len(actions)
    trainData = {}
    completeData = []

    for action in actions:
        path_to_action = os.path.join(path_to_dataset, action, category)
        for action_sample in natsorted(os.listdir(path_to_action)):
            sample = action_sample.split('.')[0]
            path_to_sample = os.path.join(path_to_action, action_sample)
            humanoid_qpos_data = np.genfromtxt(path_to_sample, delimiter=',')

            # # remove qvel from wiping data
            # humanoid_qpos_data = np.delete(humanoid_qpos_data, np.s_[35:69], axis=1)

            # remove cube pos only from co-existing data; consider goal
            humanoid_qpos_data = humanoid_qpos_data[:, :38]

            # Important data preprocessing: convert root orientation in quaternion to radians
            root_orien_quat = humanoid_qpos_data[:, 3:7]
            root_orien_radian = np.zeros_like(root_orien_quat)
            for i in range(humanoid_qpos_data.shape[0]):
                root_orien_radian[i][:3] = quat2euler(root_orien_quat[i])
            humanoid_qpos_data[:, 3:7] = root_orien_radian
            humanoid_qpos_data = np.delete(humanoid_qpos_data, 6, axis=1)  # 1 dimension reduced after conversion

            trainData[(action, sample)] = humanoid_qpos_data  # qpos data. total_dim = 35-1 = 34 (removed goal before)

            if len(completeData) == 0:
                completeData = copy.deepcopy(humanoid_qpos_data)
            else:
                completeData = np.append(completeData, humanoid_qpos_data, axis=0)

    return trainData, completeData


def load_mogaze_data(path_to_dataset: str, actions, limit):
    """
  Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

  Args
    path_to_dataset: string. directory where the data resides
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data
    category: variable to indicate whether to load test data or train data
  Returns
    trainData: dictionary with key:value
      key=(action, action_sample), value=(nxd) un-normalized data
  """
    # nactions = len(actions)
    # global path_to_human, path_to_gaze, human_pos_gaze_data
    s_human_pos_gaze_data = {}
    path_to_human = ''
    path_to_gaze = ''
    completeData = []

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
                                               30)  # 53250/30 = 1775 : 30 batches with 1775 timesteps
        s_human_pos_gaze_data = {k: v for k, v in enumerate(s_human_pos_gaze_data)}

        if config.TRAIN_MOGAZE:
            human_pos_gaze_data = np.array_split(human_pos_gaze_data,
                                                 30)  # 53250/30 = 1775 : 30 batches with 1775 timesteps
            human_pos_gaze_data = {k: v for k, v in enumerate(human_pos_gaze_data)}
            return human_pos_gaze_data, completeData

    return s_human_pos_gaze_data, completeData


def normalization_stats(completeData):
    """"
Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

Args
  completeData: nx31 matrix with data to normalize (the first 3 pos values are not normalized)
Returns
  data_mean: vector of mean used to normalize the data
  data_std: vector of standard deviation used to normalize the data
  dimensions_to_ignore: vector with dimensions not used by the model
  dimensions_to_use: vector with dimensions used by the model
"""
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def normalize_data(data, data_mean, data_std, one_hot):
    """
Normalize input data by removing unused dimensions, subtracting the mean and
dividing by the standard deviation

Args
  data: dict with each key having 34 qpos values (31 values to normalize; the first 3 pos values are not normalized)
  data_mean: vector of mean used to normalize the data
  data_std: vector of standard deviation used to normalize the data
  actions: list of strings with the encoded actions
  one_hot: whether the data comes with one-hot encoding
Returns
  data_out: the passed data matrix, but normalized
"""
    # data_len = config.MOGAZE_SIZE if config.TRAIN_MOGAZE or config.TEST_MOGAZE else config.HUMAN_SIZE
    data_len = len(data_mean)
    if not one_hot:
        # No one-hot encoding... no need to do anything special
        if isinstance(data, dict):
            for key in data.keys():
                data[key][:, :data_len] = np.divide((data[key][:, :data_len] - data_mean), data_std)
        else:
            data[..., :data_len] = np.divide((data[..., :data_len] - data_mean), data_std)

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
    for i in range(len(output_data)):
        # denormalize all data qpos & goalpos
        output_data[i] = np.multiply(output_data[i], data_std) + data_mean

        # convert root orientation from radians to quaternions
        root_orien_radian = output_data[i][:, 3:6]
        root_orien_quat = np.zeros_like(root_orien_radian)
        root_orien_quat = np.insert(root_orien_quat, 3, 0, axis=1)  # quat has 4 dimensions
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
        output_data[i] = np.multiply(output_data[i], data_std) + data_mean

    return output_data
