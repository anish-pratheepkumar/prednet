# import time
import warnings

import tensorflow as tf
from tqdm import tqdm

# from utils.evaluation.prednet_run import config
from models.prednet import motion_pred_utils
from utils.data_generation.data_utils import DataSaver, collect_qpos_hri_test
from utils.data_hri_vis.hri_utils import build_env

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

from experiments import config


def play(save_chull_data=False, save_drl_chull_data=False, terminate=False, scenarios=None, has_renderer=False):
    """
    Args:
        save_data (bool): saving data
        save_chull_data (bool): saving prediction for convex hull visualization
        save_drl_chull_data (bool): saving original for convex hull visualization
        n_episodes:
        terminate: boolean to indicate early termination
    """
    # ================================================Build Env====================================================
    workplace = build_env(has_renderer=has_renderer)
    actionables = workplace.actionables
    original_human = actionables[0]
    prediction_humans = actionables[1:]
    # ================================================Load model====================================================
    pred_model, dl_graph, sess = config.LOAD_MODEL_FN()  # (sess, config.TEST_LOAD)

    # ================================================Load Data====================================================
    if scenarios is None:
        scenarios = ["co-existing", "co-operating", "noise"]

    from utils.data_generation.data_utils import load_hri_data
    for scenario in scenarios:
        all_episodes = load_hri_data(config.HRI_DATA_PATH, action=scenario, dataset_type="test")
        # all_episodes = np.load(os.path.join(config.HRI_TEST_DATA_DIR, "all.npz"), allow_pickle=True)["all_episodes"]
        # ================================================Start QPOS Collection=============================================

        for iepisode, episode in enumerate(tqdm(all_episodes)):
            save_drl_qpos_data = []
            # save_qpos_goal_pos = np.empty(shape=[0, mogaze_size])
            normed_prev_data = []  # all_trajs[0].copy().reshape(1, -1) # step, traj
            pred_outputs = None
            chull_count = 0
            for traj in episode:  # skip 5 frames to align with trained model
                # ===================================Collect 50 steps of data===========================================
                input_data = traj.copy()
                # if config.ARCHITECTURE == "prednet":
                #     normed_pred_data = prednet_normalize_data(input_data, data_mean, data_std)
                # else:
                #     normed_pred_data = red_normalize_data(input_data, data_mean, data_std, False)
                if len(normed_prev_data) < config.SEQ_LENGTH_IN:
                    if len(normed_prev_data) == 0:
                        normed_prev_data = input_data.copy().reshape(1, -1)
                    else:
                        normed_prev_data = np.concatenate((normed_prev_data, input_data.reshape(1, -1)))
                else:
                    # ==================================Prepare the data for prediction=======================================
                    with sess.as_default():
                        pred_outputs = motion_pred_utils.predict_hri(sess, pred_model, normed_prev_data)
                    normed_prev_data = np.concatenate(
                        (normed_prev_data, input_data.reshape(1, -1)))
                    normed_prev_data = np.delete(normed_prev_data, 0, 0)
                # visualize results
                # =============Step original human====
                # original_human.step(traj)
                original_human.joint_qpos = traj
                workplace.sim.forward()
                workplace.render()
                # for actionable, traj_fut in zip(actionables[1:], all_trajs[itraj + 1::10]):
                if pred_outputs is not None:
                    # =============Step prediction humans======
                    for actionable, pred_output_ in zip(prediction_humans, pred_outputs[::2]):
                        actionable.joint_qpos = pred_output_.squeeze()
                    prediction_humans[-1].step(pred_outputs[-1].squeeze())  # ensure last human takes last prediction
                    workplace.sim.forward()
                    save_drl_qpos_data = collect_qpos_hri_test(original_human, save_drl_qpos_data)

                    chull_count += 1

                    if save_chull_data:
                        pred_data_saver = DataSaver(scenario=scenario, filename='qpos_pred_chull_viz_',
                                                    count=chull_count, filetype='.csv', data=pred_outputs,
                                                    root=config.DIR_PRED_CHULL_QPOS)
                        pred_data_saver.add_subfolder('episode_' + str(iepisode + 1))
                        pred_data_saver.save()

                        # save_qpos_chull_viz(iepisode + 1, chull_count, pred_outputs, scenario=scenario,
                        #                     root=config.DIR_PRED_CHULL_QPOS)
                workplace.render()
            if save_drl_chull_data and not terminate:
                data_saver = DataSaver(scenario=scenario, filename='qpos_drl_chull_viz_',
                                       count=iepisode + 1, filetype='.csv', data=save_drl_qpos_data,
                                       root=config.DIR_DRL_CHULL_QPOS)
                # if config.AVOID_GOAL:
                #     data_saver.add_subfolder('avoid_goal')
                data_saver.save()
            # pbar.update(1)


def run(scenarios=None, save_chull_data=True, save_drl_chull_data=True):
    global HAS_RENDERER
    # scenario_config.RENDERER = False
    HAS_RENDERER = False
    play(save_chull_data=save_chull_data,
         save_drl_chull_data=save_drl_chull_data, scenarios=scenarios, has_renderer=HAS_RENDERER)


if __name__ == '__main__':
    config.ARCHITECTURE = "red"
    config.AVOID_GOAL = False
    HAS_RENDERER = True
    config.update_experiments_dir()
    # scenarios = ["co-existing", "co-operating", "noise"]
    scenarios = ["co-operating"]  # "noise"]
    run(scenarios, save_chull_data=False, save_drl_chull_data=False)
