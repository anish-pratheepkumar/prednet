import os
import re
import statistics
import time

import numpy as np
import tensorflow as tf
# ----------- Utils -----------------
import trimesh
from natsort import natsorted
from tqdm import tqdm
from trimesh import Scene

from experiments import config
from libs.robosuite.models.mjcf_utils import new_element_from_args
# from utils.evaluation.prednet_run import config
# warnings.simplefilter(action='ignore', category=FutureWarning)
from libs.robosuite.utils.robot_utils import find_element_with_property
from utils.data_generation.data_utils import DataSaver

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HAS_RENDERER = False


def save_array_to_csv(file_path, array):
    import csv
    file = open(file_path, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(array)


def play(build_env, force_update=True, scenarios=None):
    # from utils.evaluation.mogaze_vis import apply_mogaze_settings
    # apply_mogaze_settings()
    workplace = build_env(has_renderer=HAS_RENDERER)
    actionables = workplace.actionables
    original_human = actionables[0]
    prediction_humans = actionables[1:]
    num_humanoids = len(actionables)

    def apply_sequence(qpos_data_n_humanoids):
        count = 0
        data_length = len(qpos_data_n_humanoids)
        qpos_data_limit = data_length - 2 if (data_length % 2 == 0) else data_length - 1
        if data_length <= num_humanoids:
            for iqpos in range(data_length):
                qpos = qpos_data_n_humanoids[iqpos, :]
                human_env = actionables[iqpos]
                human_env.step(qpos.squeeze())
                if iqpos == data_length - 1:
                    for i in range(num_humanoids - data_length):
                        human_env = actionables[iqpos + i + 1]
                        human_env.step(qpos.squeeze())

        else:
            for iqpos in range(0, data_length, 2):
                count += 1
                qpos = qpos_data_n_humanoids[iqpos, :]
                human_env = actionables[iqpos // 2]
                human_env.step(qpos.squeeze())
                if data_length < 25 and iqpos == qpos_data_limit:
                    for _ in range(num_humanoids - count):
                        iqpos += 2
                        human_env = actionables[iqpos // 2]
                        human_env.step(qpos.squeeze())
        # double ensure that the last qpos is assigned to the last human
        actionables[-1].step(qpos_data_n_humanoids[-1, :].squeeze())
        workplace.sim.forward()
        workplace.render()
        cvx_points = np.array([human.site_pos for human in actionables])
        return cvx_points

    def remove_mesh_asset(mesh_name):
        task = workplace.task
        mesh_asset = find_element_with_property(task.asset, "mesh", "name", mesh_name)
        task.asset.remove(mesh_asset)
        mesh_geom = find_element_with_property(task.worldbody, "geom", "name", mesh_name)
        task.worldbody.remove(mesh_geom)

    def add_new_mesh_asset(mesh_path, mesh_name, color_str=None):
        mesh_asset = new_element_from_args("mesh", **{"name": mesh_name, "file": mesh_path})
        # create geom
        color_str = color_str if color_str is not None else "0. 1. 0. 0.1"
        geom_props = {
            "name": mesh_name,  # ObjectBase.get_unique_name(mesh_name),
            "type": "mesh",
            "mesh": mesh_name,
            "rgba": color_str,
            "contype": "0",
            "conaffinity": "0",
        }
        mesh_geom = new_element_from_args(element_type="geom", **geom_props)
        task = workplace.task
        task.asset.append(mesh_asset)
        task.worldbody.append(mesh_geom)

    ms = [80, 160, 320, 400, 600, 720, 880, 1000]
    ms_frames = [2, 4, 8, 10, 15, 18, 22, 25]
    # print('start time:', time.time())

    # save cvx hull vol occupancy from qpos data as 3D meshes
    # for drl qpos data
    if scenarios is None:
        scenarios = ["co-existing", "co-operating", "noise"]
        # scenarios = [""]  # parallel run does not work due to stl generation
    pbar = tqdm(total=len(scenarios) * config.EPISODES, position=0, leave=True)

    def visualize_scene(qpos, mesh_path, mesh_name, title=None):
        # print('pred_mesh_generation completed')
        if config.VIZ_QPOS_CHULL:  # todo remove
            # mesh_name = "cvxhull_{}".format(i)
            add_new_mesh_asset(mesh_path, mesh_name)
            workplace.has_renderer = True
            workplace.sim.model.nconmax = 200
            workplace.sim.model.njmax = 200
            workplace.init_mujoco_py()
            workplace.sim.model.nconmax = 0
            workplace.sim.model.njmax = 0
            apply_sequence(qpos)
            if title is not None:
                workplace.viewer.viewer.title = title
            workplace.viewer.viewer._paused = True
            workplace.render()
            remove_mesh_asset(mesh_name)

    # n_prediction_steps = config.PREDICTION_STEPS  # replaced with number of available files
    for scenario in scenarios:
        print("=" * 50)
        print("Started VoE calculations for scenario:", scenario)
        print("=" * 50)
        used_files = []  # print(episode + 1)

        qpos_drl_chull_viz_step_dirs = natsorted(os.listdir(os.path.join(config.DIR_DRL_CHULL_QPOS, scenario)))
        for i in tqdm(qpos_drl_chull_viz_step_dirs, position=0, leave=True):
            print(i)
            episode_id = re.findall(r'\d+', i)[0]
            qpos_data = np.genfromtxt(
                os.path.join(config.DIR_DRL_CHULL_QPOS, scenario, i),
                delimiter=',')
            # qpos_data_n_humanoids = qpos_data[:config.TOTAL_STEPS, :]
            qpos_data_n_humanoids = qpos_data.copy()
            n_prediction_steps = len(qpos_data)
            # ===============================================Real Human===============================================
            # for j in range(n_prediction_steps-config.SEQ_LENGTH_OUT):
            for j in range(n_prediction_steps):
                ts_path = os.path.join(config.DIR_DRL, scenario, 'episode_{}'.format((int(episode_id))), str(j + 1))
                os.makedirs(ts_path, exist_ok=True)
                mesh_name = 'cvxhull_drl_step_{}.stl'.format(j + 1)
                mesh_path = os.path.join(ts_path, mesh_name)
                if not force_update and os.path.isfile(mesh_path) and os.path.exists(mesh_path):
                    qpos_data_ts = None
                else:
                    # create mesh file for the complete prediction step
                    qpos_data_ts = qpos_data_n_humanoids[j:config.SEQ_LENGTH_OUT + j, :]
                    cvx_points = apply_sequence(qpos_data_ts)

                    mesh = trimesh.convex.convex_hull(cvx_points.reshape(-1, 3))
                    mesh.export(mesh_path)
                # if j>3:
                #     visualize_scene(qpos_data_ts, mesh_path, mesh_name=mesh_name)

                # create mesh files for multiple prediction time steps
                for k in ms_frames:
                    mesh_name = 'cvxhull_drl_{}.stl'.format(ms[ms_frames.index(k)])
                    mesh_path = os.path.join(ts_path, mesh_name)
                    if not force_update and os.path.isfile(mesh_path) and os.path.exists(mesh_path):
                        continue
                    else:
                        if qpos_data_ts is None:
                            qpos_data_ts = qpos_data_n_humanoids[j:config.SEQ_LENGTH_OUT + j, :]
                        cvx_points = apply_sequence(qpos_data_ts[:k])
                        mesh = trimesh.convex.convex_hull(cvx_points.reshape(-1, 3))
                        mesh.export(mesh_path)
                # visualize_scene(qpos_data_ts, mesh_path, mesh_name=mesh_name)
        # print('drl_mesh_generation completed')
        # for pred qpos data
        qpos_pred_chull_viz_step_dirs = natsorted(os.listdir(os.path.join(config.DIR_PRED_CHULL_QPOS, scenario)))
        n_episodes = len(qpos_pred_chull_viz_step_dirs)
        for episode_dir in tqdm(qpos_pred_chull_viz_step_dirs, position=0, leave=True):
            # print(episode_dir)
            os.makedirs(os.path.join(config.DIR_PRED, scenario, episode_dir), exist_ok=True)
            episode_path = os.path.join(config.DIR_PRED_CHULL_QPOS, scenario, episode_dir)
            n_steps = len(natsorted(os.listdir(episode_path)))
            for i in natsorted(os.listdir(episode_path)):
                steps = re.findall(r'\d+', i)[0]
                if int(steps) <= n_steps:
                    ts_path = os.path.join(config.DIR_PRED, scenario, episode_dir, str(steps))
                    os.makedirs(ts_path, exist_ok=True)
                    mesh_path = os.path.join(ts_path, 'cvxhull_pred_step_{}.stl').format(steps)
                    if not force_update and os.path.isfile(mesh_path) and os.path.exists(mesh_path):
                        qpos_data = None
                    else:
                        qpos_data = np.genfromtxt(os.path.join(episode_path, i), delimiter=',')

                        # create mesh file for the complete prediction step
                        qpos_data_ts = qpos_data  # 25x35
                        cvx_points = apply_sequence(qpos_data_ts)

                        mesh = trimesh.convex.convex_hull(cvx_points.reshape(-1, 3))
                        mesh.export(mesh_path)

                    # create mesh files for multiple prediction time steps
                    for k in ms_frames:
                        mesh_path = os.path.join(ts_path, 'cvxhull_pred_{}.stl').format(ms[ms_frames.index(k)])
                        if not force_update and os.path.isfile(mesh_path) and os.path.exists(mesh_path):
                            continue
                        else:
                            if qpos_data is None:
                                qpos_data = np.genfromtxt(os.path.join(episode_path, i), delimiter=',')
                                qpos_data_ts = qpos_data  # 25x35
                            cvx_points = apply_sequence(qpos_data_ts[:k])
                            mesh = trimesh.convex.convex_hull(cvx_points.reshape(-1, 3))
                            mesh.export(mesh_path)

        # compute VOE error and viz actual, prediction and intersection 3D cvx hulls
        # print('starting intersection mesh generation and VOE error computation')
        voc_ms_error = np.empty(shape=[0, 8])
        voc_step_error = None
        voc_step_error_list = []
        # 30 episodes each with 325 prediction steps ( 1 prediction step = 25 ts)

        for episode in range(n_episodes):
            pred_ts_dir_all_steps = os.path.join(config.DIR_PRED, scenario, 'episode_' + str(episode + 1))
            drl_ts_dir_all_steps = os.path.join(config.DIR_DRL, scenario, 'episode_' + str(episode + 1))
            n_prediction_step_id = min(
                len(natsorted(os.listdir(pred_ts_dir_all_steps))),
                len(natsorted(os.listdir(drl_ts_dir_all_steps)))
            )
            pbar_2 = tqdm(total=n_prediction_step_id, position=0, leave=True)
            for step_id in tqdm(range(n_prediction_step_id), position=0, leave=True):
                voc_ms_error_list = []
                mesh_drl, mesh_name_drl, mesh_path_drl, mesh_pred, mesh_name_pred, mesh_path_pred = ([] for _ in
                                                                                                     range(6))
                pred_ts_dir = os.path.join(pred_ts_dir_all_steps, str(step_id + 1))
                drl_ts_dir = os.path.join(drl_ts_dir_all_steps, str(step_id + 1))

                # load prediction and DRL cvx hull vol occupancy meshes
                step_mesh_drl_pred_paths = natsorted(os.listdir(pred_ts_dir))
                for i in step_mesh_drl_pred_paths:
                    mesh_pred.append(trimesh.load_mesh(os.path.join(pred_ts_dir, i)))
                    mesh_path_pred.append(os.path.join(pred_ts_dir, i))
                    mesh_name_pred.append(i)

                step_mesh_drl_paths = natsorted(os.listdir(drl_ts_dir))
                for i in step_mesh_drl_paths:
                    mesh_drl.append(trimesh.load_mesh(os.path.join(drl_ts_dir, i)))
                    mesh_path_drl.append(os.path.join(drl_ts_dir, i))
                    mesh_name_drl.append(i)

                for i in range(len(mesh_drl)):
                    intersection_ts_dir = os.path.join(config.DIR_INTERSECTION, scenario, 'episode_' + str(episode + 1),
                                                       str(step_id + 1))
                    os.makedirs(intersection_ts_dir, exist_ok=True)
                    mesh_path_intersection = os.path.join(intersection_ts_dir, 'cvxhull_intersection_{}_{}.stl')
                    mesh_name_intersection = "cvxhull_intersection_{}_{}"
                    # find intersection between prediction and DRL vol occupancy regions
                    # if os.path.isfile(mesh_path) and os.path.exists(mesh_path):
                    #     continue

                    # find intersection of step prediction
                    if 'step' in mesh_name_drl[i]:
                        # add the intersection mesh
                        temp_path = mesh_path_intersection.format('step', str(step_id + 1))
                        if not force_update and os.path.isfile(temp_path) and os.path.exists(temp_path):
                            intersection_mesh = trimesh.load_mesh(temp_path)
                        else:
                            intersection_mesh = mesh_drl[i].intersection(mesh_pred[i])
                            intersection_mesh.export(temp_path)
                        temp_name = mesh_name_intersection.format('step', str(step_id + 1))
                        if isinstance(intersection_mesh, Scene):
                            # print('there is no intersection region')
                            intersection_volume = 0.
                        else:
                            if intersection_mesh.vertices.size is not 0:
                                intersection_volume = intersection_mesh.volume
                                # add_new_mesh_asset(temp_path, temp_name, color_str="1. 0. 0. 0.2")
                            else:
                                intersection_volume = 0.
                                # print('there is no intersection region')

                        # compute VOC error
                        real_volume = mesh_drl[i].volume
                        voc_step_error = abs((intersection_volume - real_volume) / real_volume) * 100
                        pbar_2.set_description("VOE_STEP: {:.2f}".format(voc_step_error))
                        used_files.append((scenario, episode, step_id, i,
                                           os.path.join(drl_ts_dir, step_mesh_drl_paths[i]),
                                           os.path.join(pred_ts_dir, step_mesh_drl_pred_paths[i])
                                           , temp_path))

                    else:
                        # add the intersection mesh
                        temp_path = mesh_path_intersection.format(str(ms[i]), 'ms')
                        if not force_update and os.path.isfile(temp_path) and os.path.exists(temp_path):
                            intersection_mesh = trimesh.load_mesh(temp_path)
                        else:
                            try:
                                intersection_mesh = mesh_drl[i].intersection(mesh_pred[i])
                                intersection_mesh.export(temp_path)
                            except:
                                intersection_mesh = None
                                pass
                                print(str(step_id + 1), i, "is failure")
                                intersection_volume = 0.
                        used_files.append((scenario, episode, step_id, i,
                                           os.path.join(drl_ts_dir, step_mesh_drl_paths[i]),
                                           os.path.join(pred_ts_dir, step_mesh_drl_pred_paths[i])
                                           , temp_path))
                        temp_name = mesh_name_intersection.format(str(ms[i]), 'ms')
                        if isinstance(intersection_mesh, Scene):
                            # print('there is no intersection region')
                            intersection_volume = 0.
                        else:
                            if intersection_mesh.vertices.size is not 0:
                                intersection_volume = intersection_mesh.volume
                                # add_new_mesh_asset(temp_path, temp_name, color_str="1. 0. 0. 0.2")
                            else:
                                intersection_volume = 0.
                                # print('there is no intersection region')

                        # compute VOC error
                        try:
                            real_volume = mesh_drl[i].volume
                        except:
                            print(str(step_id + 1), i, "is failure")

                        voc_error = abs((intersection_volume - real_volume) / real_volume) * 100
                        voc_ms_error_list.append(voc_error)
                        pbar_2.set_description("VOE_MS: {:.2f}".format(voc_error))
                    pbar_2.update(1)
                    if config.VIZ_VOE_CHULL and i == len(mesh_drl) - 1:
                        # apply qpos
                        episode_path = os.path.join(config.DIR_PRED_CHULL_QPOS, scenario,
                                                    "episode_{}".format(episode + 1))
                        pred_qpos_path = os.path.join(episode_path, "qpos_pred_chull_viz_{}.csv".format(step_id + 1))
                        pred_qpos_data = np.genfromtxt(pred_qpos_path, delimiter=',')
                        real_episode_path = os.path.join(config.DIR_DRL_CHULL_QPOS, scenario)
                        real_episode_path = os.path.join(real_episode_path,
                                                         "qpos_drl_chull_viz_{}.csv".format(episode + 1))
                        real_qpos_data = np.genfromtxt(real_episode_path, delimiter=',')
                        # # add drl & pred mesh
                        visualize_scene(real_qpos_data[
                                        step_id:step_id + config.SEQ_LENGTH_OUT,
                                        :]
                                        , mesh_path_drl[i], mesh_name_drl[i])
                        visualize_scene(pred_qpos_data, mesh_path_pred[i], mesh_name_pred[i])
                voc_step_error_list.append(voc_step_error)
                voc_ms_error = np.vstack((voc_ms_error, np.array(voc_ms_error_list).reshape(1, 8)))
            pbar.update(1)
        voc_avg_step_error = statistics.mean(voc_step_error_list)
        print('voc_avg_step_error = ', voc_avg_step_error)
        voc_avg_ms_error = np.mean(voc_ms_error, 0).squeeze().tolist()
        voc_std_ms_error = np.std(voc_ms_error, 0).squeeze().tolist()

        voe_dir = os.path.join(config.CHULL_BASE_DIR, "voe_result")
        data_saver = DataSaver(scenario=scenario, filename='voc_ms_error',
                               count=3, filetype='.csv', data=voc_ms_error, root=voe_dir)
        # if config.AVOID_GOAL:
        #     data_saver.add_subfolder('avoid_goal')
        data_saver.save()

        voc_step_error_np = np.array(voc_step_error_list)
        data_saver_1 = DataSaver(scenario=scenario, filename='voc_step_error',
                                 count=3, filetype='.csv', data=voc_step_error_np, root=voe_dir)
        # if config.AVOID_GOAL:
        #     data_saver_1.add_subfolder('avoid_goal')
        data_saver_1.save()
        try:
            save_array_to_csv(os.path.splitext(data_saver_1.get_save_path())[0] + "_debug_gen_files.csv",
                              used_files)
        except:
            pass
        # if config.VIZ_VOE_CHULL:
        #     workplace.has_renderer = True
        #     workplace.init_mujoco_py()
        #     workplace.viewer.viewer._paused = True
        #     workplace.render()
        #     workplace.has_renderer = False
        print()
        print("{0: <16} |".format("prediction_step"), end="")
        for i in ms:
            print(" {0:5d} |".format(i), end="")
        print()
        # 1 time step = 40 ms
        print("{0: <16} |".format('voc_error ± std (%)'), end="")
        for voc_error, voc_std_error in zip(voc_avg_ms_error, voc_std_ms_error):
            print(" {0:.3f} ± {1:.2f}|".format(voc_error, voc_std_error / 2), end="")
        print()
        print('End time:', time.time())
    pbar.close()


if __name__ == '__main__':
    # HAS_RENDERER = True
    # config.VIZ_VOE_CHULL = True
    # config.ARCHITECTURE = "prednet"
    # apply_mogaze_settings()
    # from utils.data_mogaze_vis.mogaze_utils import build_env
    config.ARCHITECTURE = "prednet"
    config.AVOID_GOAL = False
    config.VIZ_VOE_CHULL = False
    config.update_experiments_dir()
    scenarios = ["co-existing", "co-operating", "noise"]
    from utils.data_hri_vis.hri_utils import build_env

    config.VIZ_QPOS_CHULL = True
    play(build_env, force_update=False, scenarios=scenarios)
