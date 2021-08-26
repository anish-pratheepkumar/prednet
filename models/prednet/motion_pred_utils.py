import os

import numpy as np

from experiments import config
from utils.data_generation import data_utils

is_initialized = False
data_mean, data_std = None, None


def get_stats():
    global is_initialized, data_mean, data_std
    if is_initialized:
        return data_mean, data_std
    else:
        # if config.ARCHITECTURE is "red":
        #     from models.red.HRI_Scenario import HRI_translate
        #     from models.red.HRI_Scenario.HRI_translate import FLAGS
        #     _, _, data_mean, data_std = HRI_translate.read_qpos_data(FLAGS.seq_length_in,
        #                                                              FLAGS.seq_length_out,
        #                                                              FLAGS.data_dir,
        #                                                              not FLAGS.omit_one_hot)
        # else:
        data_mean = np.genfromtxt(os.path.join(config.NORM_STAT_DIR, 'data_mean.csv'), delimiter=',')
        data_std = np.genfromtxt(os.path.join(config.NORM_STAT_DIR, 'data_std.csv'), delimiter=',')
        is_initialized = True
        return data_mean, data_std


def predict_hri(sess, pred_model, pred_data):
    """make predictions using PredNet / RED model"""

    # Load data_mean and std_dev
    # data_mean = np.genfromtxt(config.DATA_MEAN_FILE, delimiter=',')
    # data_std = np.genfromtxt(config.DATA_STD_FILE, delimiter=',')

    # data preprocessing
    # convert root orientation from quat to Euler
    # -------------------TODO----------------
    pred_data_radian = data_utils.load_qpos_data(pred_data)
    data_mean, data_std = get_stats()

    # normalize the data to predict
    pred_data_norm = data_utils.normalize_data(pred_data_radian, data_mean, data_std)

    if config.ARCHITECTURE ==  "red":
        encoder_inputs = np.zeros((1, config.SEQ_LENGTH_IN - 1, config.QPOS_SIZE), dtype=float)
        decoder_inputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE), dtype=float)
        decoder_outputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE-config.GOAL_SIZE), dtype=float)
    else:
        encoder_inputs = np.zeros((1, config.SEQ_LENGTH_IN - 1, config.QPOS_SIZE), dtype=float)
        decoder_inputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE), dtype=float)
        decoder_outputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE), dtype=float)

    # considering only last 50 qpos sequences of the collected data for prediction
    # get the number of sequences in collected data
    n = pred_data_norm.shape[0]
    # loading encoder inputs -> 49 sequences
    encoder_inputs[0, :, 0:config.QPOS_SIZE] = pred_data_norm[(n - 1) - 49:n - 1, :]
    # loading the first seed value for decoding; this is the last sequence(50th) of qpos input;
    # remaining values can be zero since we are using sampling based loss
    decoder_inputs[0, 0, 0:config.QPOS_SIZE] = pred_data_norm[-1, :]

    forward_only = True
    outputs = pred_model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs[:, :, :34], forward_only, pred=True)
    outputs = np.squeeze(np.stack(outputs, axis=0))

    # data post processing: denormalize root orientation and joint angle rotations
    # then convert root orientations to quat from Euler
    final_outputs = data_utils.post_process(outputs, data_mean, data_std)

    return final_outputs
