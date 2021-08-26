import os
import sys

project_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

# -------------------------
import os
import random

import numpy as np
import tensorflow as tf

from experiments import config
from models.prednet.data_utils import normalize_data as prednet_normalize_data, post_process as prednet_post_process
from models.red.RED_data_utils import normalize_data as red_normalize_data, post_process as red_post_process
from utils.data_hri_vis.hri_utils import build_env


def apply_hri_settings(test=True):
    if test:
        config.TEST_MOGAZE = False
        # config.ITERATIONS =
        # config.TEST_LOAD = 6000
        # config.ARCHITECTURE = "prednet"
        config.ACTION = "combined"
        # config.USE_CPU = True
        if config.TEST_LOAD <= 0:
            raise (ValueError, "Must give an iteration to read parameters from")
    config.update_experiments_dir()


def predict_test_hri():
    # ================================================Load Model====================================================
    print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
    # sampling     = True
    pred_model, dl_graph, sess = config.LOAD_MODEL_FN()  # (sess, config.TEST_LOAD)
    print("Model created")
    # ================================================Workplace Env====================================================
    workplace = build_env()
    actionables = workplace.actionables
    n_humans = len(workplace.actionables)
    # ================================================Model Params=====================================================
    from utils.data_generation.data_utils import load_hri_data
    all_episodes = load_hri_data(config.HRI_DATA_PATH, action="combined", dataset_type="test")

    with sess.as_default():
        tf.set_random_seed(0)
        random.seed(0)
        np.random.seed(0)

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Load Norm Stats
        # --------------------------------------------------------------------------------------------------------------
        # Load data_mean and std_dev
        qpos_size = config.QPOS_SIZE
        data_mean = np.genfromtxt(os.path.join(config.NORM_STAT_DIR, 'data_mean.csv'), delimiter=',')
        data_std = np.genfromtxt(os.path.join(config.NORM_STAT_DIR, 'data_std.csv'), delimiter=',')

        normed_prev_data = []  # all_trajs[0].copy().reshape(1, -1) # step, traj
        pred_outputs = None
        # --------------------------------------------------------------------------------------------------------------
        #                                                 Start prediction
        # --------------------------------------------------------------------------------------------------------------
        for episode in all_episodes:
            for traj in episode:
                # Normalize -- subtract mean of train data, divide by stdev of train data
                input_data = traj.copy()
                if config.ARCHITECTURE == "prednet":
                    normed_pred_data = prednet_normalize_data(input_data, data_mean, data_std)
                else:
                    normed_pred_data = red_normalize_data(input_data, data_mean, data_std, False)

                if len(normed_prev_data) == config.SEQ_LENGTH_IN:
                    # ==================================Prepare the data for prediction=======================================

                    if config.ARCHITECTURE == "red":
                        encoder_inputs = np.zeros((1, config.SEQ_LENGTH_IN - 1, config.QPOS_SIZE), dtype=float)
                        decoder_inputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE), dtype=float)
                        decoder_outputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE - config.GOAL_SIZE),
                                                   dtype=float)
                    else:
                        encoder_inputs = np.zeros((1, config.SEQ_LENGTH_IN - 1, config.QPOS_SIZE), dtype=float)
                        decoder_inputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE), dtype=float)
                        decoder_outputs = np.zeros((1, config.SEQ_LENGTH_OUT, config.QPOS_SIZE), dtype=float)

                    encoder_inputs = normed_prev_data[0:config.SEQ_LENGTH_IN - 1, :encoder_inputs.shape[2]]
                    encoder_inputs = encoder_inputs.reshape((-1, *encoder_inputs.shape))  # input at t-49: t
                    decoder_inputs[0, 0, :] = normed_pred_data[:decoder_inputs.shape[2]]
                    # decoder_inputs = normed_pred_data.reshape((1, 1, -1))

                    # ==================================Predict=======================================
                    normed_pred_outputs = pred_model.step(sess, encoder_inputs, decoder_inputs,
                                                          decoder_outputs[:, :, :decoder_outputs.shape[2]],
                                                          forward_only=True, pred=True)
                    if config.ARCHITECTURE == "prednet":
                        pred_outputs = prednet_post_process(normed_pred_outputs, data_mean, data_std)
                    else:
                        # todo again problem with normalization
                        # pred_outputs = prednet_post_process_mogaze(normed_pred_outputs, data_mean, data_std)
                        pred_outputs = red_post_process(normed_pred_outputs, data_mean, data_std)

                    normed_prev_data = np.concatenate(
                        (normed_prev_data, normed_pred_data.reshape(1, -1)))  # todo define axis
                    normed_prev_data = np.delete(normed_prev_data, 0, 0)
                else:
                    # ==================================Collect more samples=======================================
                    if len(normed_prev_data) == 0:
                        normed_prev_data = normed_pred_data.copy().reshape(1, -1)
                    else:
                        normed_prev_data = np.concatenate((normed_prev_data, normed_pred_data.reshape(1, -1)))

                # visualize results
                actionables[0].step(traj)
                # for actionable, traj_fut in zip(actionables[1:], all_trajs[itraj + 1::10]):
                if pred_outputs is not None:
                    for actionable, pred_output_ in zip(actionables[1:], pred_outputs[::2]):
                        actionable.step(pred_output_.squeeze())
                workplace.sim.forward()
                # workplace.viewer.viewer._paused = True
                workplace.render()
                # save predictions
                # pred_outputs

                # save sites for cvxhull calculation


if __name__ == '__main__':
    config.ARCHITECTURE = "red"
    # config.ARCHITECTURE = "prednet"
    config.AVOID_GOAL = False
    apply_hri_settings()
    predict_test_hri()
