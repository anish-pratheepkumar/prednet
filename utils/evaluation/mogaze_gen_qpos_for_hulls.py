# import time
import warnings

import tensorflow as tf
from tqdm import tqdm
from utils.data_generation.data_utils import DataSaver, collect_qpos_goal_pos_mogaze, save_qpos_chull_viz

# from utils.evaluation.prednet_run import config
from utils.evaluation.mogaze_vis import apply_mogaze_settings

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from models.prednet.data_utils import post_process_mogaze as prednet_post_process_mogaze
from models.red.RED_data_utils import post_process_mogaze as red_post_process_mogaze
import os

import numpy as np

from experiments import config
from models.prednet.data_utils import normalize_data as prednet_normalize_data
from models.red.RED_data_utils import normalize_data as red_normalize_data
from utils.data_mogaze_vis.mogaze_utils import build_env


def play(save_chull_data=False, save_drl_chull_data=False, terminate=False):
    """
    Args:
        save_data (bool): saving data
        save_chull_data (bool): saving prediction for convex hull visualization
        save_drl_chull_data (bool): saving original for convex hull visualization
        n_episodes:
        terminate: boolean to indicate early termination
    """
    # from utils.evaluation.mogaze_vis import apply_mogaze_settings
    # apply_mogaze_settings()
    mogaze_size = config.MOGAZE_SIZE if config.AVOID_GOAL and config.ARCHITECTURE == "prednet" else config.MOGAZE_SIZE + config.GOAL_SIZE
    real_mogaze_size = config.MOGAZE_SIZE
    # ================================================Load model====================================================
    # if config.ARCHITECTURE is "red":
    #     from models.red.HRI_Scenario import HRI_translate as RED
    #     pred_model, dl_graph, sess = RED.load_model()
    # else:
    #     from models.prednet.prednet_model import PredNet
    #     pred_model, dl_graph, sess = PredNet.load_model()
    pred_model, dl_graph, sess = config.LOAD_MODEL_FN()  # (sess, config.TEST_LOAD)

    # --------------------------------------------------------------------------------------------------------------
    #                                                 Load Model
    # --------------------------------------------------------------------------------------------------------------
    # print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
    # pred_model = create_model(sess, config.TEST_LOAD)
    # print("Model created")
    # ================================================Load Data====================================================
    from utils.data_generation.data_utils import load_mogaze_data
    real_data, all_trajs = load_mogaze_data(config.MOGAZE_DATA_DIR, actions=['p2_1'], limit=19500)
    # Load data_mean and std_dev
    data_mean = np.genfromtxt(os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_mean.csv'), delimiter=',')
    data_std = np.genfromtxt(os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_std.csv'), delimiter=',')

    # ================================================Build Env====================================================
    workplace = build_env(has_renderer=HAS_RENDERER)
    actionables = workplace.actionables
    original_human = actionables[0]
    prediction_humans = actionables[1:]
    # ================================================Start QPOS Collection=============================================
    episodes = 0
    chull_count = 0

    save_drl_qpos_data = []
    # save_qpos_goal_pos = np.empty(shape=[0, mogaze_size])
    normed_prev_data = []  # all_trajs[0].copy().reshape(1, -1) # step, traj
    pred_outputs = None
    for traj in tqdm(all_trajs[::5]):  # skip 5 frames to align with trained model
        # ===================================Collect 50 steps of data===========================================
        input_data = traj.copy()
        if config.ARCHITECTURE == "prednet":
            normed_pred_data = prednet_normalize_data(input_data, data_mean, data_std)
        else:
            normed_pred_data = red_normalize_data(input_data, data_mean, data_std, False)
        if len(normed_prev_data) < config.SEQ_LENGTH_IN:
            if len(normed_prev_data) == 0:
                normed_prev_data = normed_pred_data.copy().reshape(1, -1)
            else:
                normed_prev_data = np.concatenate((normed_prev_data, normed_pred_data.reshape(1, -1)))
        else:
            # ==================================Prepare the data for prediction=======================================

            encoder_inputs = normed_prev_data[0:config.SEQ_LENGTH_IN - 1, :mogaze_size]
            encoder_inputs = encoder_inputs.reshape((-1, *encoder_inputs.shape))  # input at t-49: t
            decoder_inputs = np.zeros((1, config.SEQ_LENGTH_OUT, mogaze_size), dtype=float)
            decoder_inputs[0, 0, :] = normed_pred_data[:mogaze_size]
            decoder_outputs = np.zeros((1, config.SEQ_LENGTH_OUT, real_mogaze_size), dtype=float)

            # ==================================Predict=======================================
            normed_pred_outputs = pred_model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs[:, :, :66],
                                                  forward_only=True, pred=True)
            if config.ARCHITECTURE == "prednet":
                pred_outputs = prednet_post_process_mogaze(normed_pred_outputs, data_mean, data_std)
            else:
                pred_outputs = red_post_process_mogaze(normed_pred_outputs, data_mean, data_std)

            normed_prev_data = np.concatenate(
                (normed_prev_data, normed_pred_data.reshape(1, -1)))  # todo define axis
            normed_prev_data = np.delete(normed_prev_data, 0, 0)
        # visualize results
        # =============Step original human====
        original_human.step(traj)
        workplace.sim.forward()
        workplace.render()
        # for actionable, traj_fut in zip(actionables[1:], all_trajs[itraj + 1::10]):
        if pred_outputs is not None:
            # =============Step prediction humans======
            for actionable, pred_output_ in zip(prediction_humans, pred_outputs[::2]):
                actionable.step(pred_output_.squeeze())
            prediction_humans[-1].step(pred_outputs[-1].squeeze())  # ensure last human takes last prediction

            workplace.sim.forward()
            save_drl_qpos_data = collect_qpos_goal_pos_mogaze(original_human, save_drl_qpos_data, chull=True)

            chull_count += 1

            if save_chull_data:
                save_qpos_chull_viz(episodes + 1, chull_count, pred_outputs, scenario='',
                                    root=config.DIR_PRED_CHULL_QPOS)
                # pred_data_saver = DataSaver(scenario="", filename='qpos_pred_chull_viz_',
                #                             count=episodes + 1, filetype='.csv', data=pred_outputs,
                #                             root=config.DIR_PRED_CHULL_QPOS)
                # pred_data_saver.save()
        workplace.render()

    if save_drl_chull_data and not terminate:
        data_saver = DataSaver(scenario='', filename='qpos_drl_chull_viz_',
                               count=episodes + 1, filetype='.csv', data=save_drl_qpos_data,
                               root=config.DIR_DRL_CHULL_QPOS)
        if config.AVOID_GOAL:
            data_saver.add_subfolder('avoid_goal')
        data_saver.save()


def run():
    global HAS_RENDERER
    HAS_RENDERER = False
    play(save_chull_data=True,
         save_drl_chull_data=True)


if __name__ == '__main__':
    save_chull_data = True  # save qpos data of dl and drl human for chull visualization of prediction
    save_drl_chull_data = True  # save qpos data for drl human to calculate vol. occ. error
    HAS_RENDERER = False
    config.ARCHITECTURE = "prednet"
    apply_mogaze_settings()
    play(save_chull_data,
         save_drl_chull_data)
