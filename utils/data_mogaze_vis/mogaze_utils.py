import os
import sys

import click

project_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir

from collections import OrderedDict

import numpy as np
from humoro.trajectory import Trajectory
from scipy.spatial.transform import Rotation as R

import utils
import utils.data_generation.data_utils as data_utils
from experiments import config
from libs.robosuite.environments.base import MujocoInitEnv
from libs.robosuite.environments.workplaces import WorkplaceEnv
from libs.robosuite.models.arenas import EmptyArena
from libs.robosuite.models.mjcf_utils import string_to_array, new_element, array_to_string
from libs.robosuite.models.objects.objects import EmptyObject, Goal
from libs.robosuite.models.tasks import Task


# -------------------------------------------------------------------------------------------------------------------
#                                                   Model Staff
# -------------------------------------------------------------------------------------------------------------------

def run_test_mogaze():
    import copy
    import os
    import random

    import numpy as np

    from models.prednet import data_utils
    from experiments import config
    from models.prednet.train_prednet import check_action, create_model, print_results
    import tensorflow as tf

    config.TEST_MOGAZE = True
    if config.TEST_LOAD <= 0:
        raise (ValueError, "Must give an iteration to read parameters from")

    check_action(config.ACTION)

    # Use the CPU if asked to
    device_count = {"GPU": 0} if config.USE_CPU else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(device_count=device_count)) as sess:
        tf.set_random_seed(0)
        random.seed(0)
        np.random.seed(0)

        # === Create the model ===
        print("Creating %d layers of %d units." % (config.NUM_LAYERS, config.SIZE))
        # sampling     = True
        pred_model = create_model(sess, config.TEST_LOAD)
        print("Model created")

        # Load data_mean and std_dev
        data_mean = np.genfromtxt(os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_mean.csv'), delimiter=',')
        data_std = np.genfromtxt(os.path.join(config.MOGAZE_NORM_STAT_DIR, 'data_std.csv'), delimiter=',')

        # load real data as ground truth
        real_data, _ = utils.data_generation.data_utils.load_mogaze_data(config.MOGAZE_DATA_DIR, actions=['p2_1'],
                                                                         limit=19500)

        # load pred data for prediction
        pred_data = copy.deepcopy(real_data)

        # Normalize -- subtract mean of train data, divide by stdev of train data
        normed_pred_data = data_utils.normalize_data(pred_data, data_mean, data_std)

        # Make prediction
        mogaze_size = config.MOGAZE_SIZE if config.AVOID_GOAL else config.MOGAZE_SIZE + config.GOAL_SIZE  # dim in one time step of qpos data
        real_mogaze_size = config.MOGAZE_SIZE
        encoder_inputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_IN - 1, mogaze_size), dtype=float)
        decoder_inputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, mogaze_size), dtype=float)
        decoder_outputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, mogaze_size), dtype=float)
        real_decoder_outputs = np.zeros((config.BATCH_SIZE, config.SEQ_LENGTH_OUT, real_mogaze_size), dtype=float)

        sub_batches = 1  # config.MOGAZE_TEST_SUB_BATCH_SIZE
        pred_test_loss = 0
        pred_test_ms_loss = 0
        # send 50 data in each loop for prediction
        batch_keys = list(normed_pred_data.keys())

        for i in range(config.BATCH_SIZE):
            # load sub batch to predict
            the_key = batch_keys[i]

            # Select the data around the sampled points
            data_sel = normed_pred_data[the_key][config.SEQ_LENGTH_IN: config.SEQ_LENGTH_IN + config.SEQ_LENGTH_IN, :]
            # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
            encoder_inputs[i, :, 0:mogaze_size] = data_sel[0:config.SEQ_LENGTH_IN - 1, :]
            decoder_inputs[i, 0, 0:mogaze_size] = data_sel[-1, :]

        for i in range(config.BATCH_SIZE):
            # load sub batch real data corresponding to prediction (ground truths)
            the_key = batch_keys[i]

            # Select the data around the sampled points
            data_sel = real_data[the_key][
                       config.SEQ_LENGTH_IN + config.SEQ_LENGTH_IN: config.SEQ_LENGTH_IN + config.SEQ_LENGTH_IN + config.SEQ_LENGTH_OUT,
                       :]
            # Add the data to each batch i.e, i will vary from 0 to 29 => 30 batches
            real_decoder_outputs[i, :, :] = data_sel[:, :66]

        pred_outputs = pred_model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs[:, :, :66],
                                       forward_only=True, pred=True)
        final_pred_outputs = data_utils.post_process_mogaze(pred_outputs, data_mean, data_std)
        final_pred_outputs = np.stack(final_pred_outputs, axis=1)  # modify dimension to dimensions to 30x25x66
        # real decoder output dimension is 30x25x35

        # pred_test_loss += np.sqrt(np.mean(np.square(np.subtract(real_decoder_outputs, final_pred_outputs))))
        # pred_test_ms_loss += np.sqrt(np.mean(np.square(np.subtract(real_decoder_outputs, final_pred_outputs)), axis=(0,2)))  # loss at each timestep(1TS = 40ms)

        # MAE
        pred_test_loss += np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)))
        pred_test_ms_loss += np.mean(np.absolute(np.subtract(real_decoder_outputs, final_pred_outputs)),
                                     axis=(0, 2))

        avg_pred_test_loss = pred_test_loss / sub_batches
        print('total prediction test loss : {}'.format(avg_pred_test_loss))

        avg_pred_test_ms_loss = pred_test_ms_loss / sub_batches

        print_results(avg_pred_test_ms_loss)


# -------------------------------------------------------------------------------------------------------------------
#                                                   Utils
# -------------------------------------------------------------------------------------------------------------------
def get_parent_node(root, child):
    """
    Return parent element of a child
    Args:
        model (XMLElement):
        child (XMLElement):

    Returns (XMLElement): direct parent node of child

    """
    parent_map = {c: p for p in root.iter() for c in p}
    return parent_map.get(child)


def gen_vhull_sites(human):
    # def sample_spherical(npoints, sphere_radius, ndim=3):
    #     vec = np.random.randn(ndim, npoints)
    #     vec /= np.linalg.norm(vec, axis=0) * sphere_radius
    #     return vec
    def sample_points_cylinder_surf(radius, max_height, n_points=10):
        rand_theta = np.random.uniform(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)
        rand_z = np.random.uniform(0., max_height, n_points).reshape(-1, 1)
        rand_points = np.concatenate([radius * np.cos(rand_theta),
                                      radius * np.sin(rand_theta),
                                      rand_z], axis=1)
        return rand_points

    def sample_points_sphere_surf(sphere_radius, n_points=20):
        rand_gamma = np.random.uniform(0., np.pi, n_points).reshape(-1, 1)
        rand_theta = np.random.uniform(-2 * np.pi, 2 * np.pi, n_points).reshape(-1, 1)

        rand_points = np.concatenate([sphere_radius * np.sin(rand_gamma) * np.cos(rand_theta),
                                      sphere_radius * np.sin(rand_gamma) * np.sin(rand_theta),
                                      sphere_radius * np.cos(rand_gamma)
                                      ], axis=1)
        return rand_points

    # generate sites on the surface of the geoms
    all_geoms = human.worldbody.findall(".//geom")
    for i, geom in enumerate(all_geoms):
        geom_type = geom.get("type", "sphere")
        geom_size = string_to_array(geom.get("size"))
        geom_quat = string_to_array(geom.get("quat", "1 0 0 0"))
        geom_pos = string_to_array(geom.get("pos", "0 0 0"))
        if geom_type == "sphere":
            # sphere_radius = geom_size[0]
            # rand_theta = np.random.uniform(0., 2 * np.pi, 100)
            # rand_points = np.vstack([sphere_radius * np.cos(rand_theta), sphere_radius * np.sin(rand_theta),
            #                     np.ones_like(rand_theta) * sphere_radius])
            rand_points = sample_points_sphere_surf(geom_size[0])
            # tips of sphere
            # add site at sphere radius
            # rand_points = sample_spherical(100, geom_size[0]) + geom_pos
        elif geom_type == "box":
            # add site at x,y,z combinations
            max_dim = np.max(geom_size)
            rand_points = sample_points_sphere_surf(max_dim)

        elif geom_type == "cylinder":
            # add site at multiple height, each with multiple radius
            rand_points = sample_points_cylinder_surf(geom_size[0], geom_size[1])
        else:
            print(geom)
        goem_parent_body = get_parent_node(human.worldbody, geom)
        geom_r = R.from_quat((*geom_quat[1:], geom_quat[0]))
        for ip, point in enumerate(rand_points):
            transformed_point = geom_r.apply(point) + geom_pos
            site = new_element("site", name="{}:site_{}_{}".format(human.object_id, i, ip), rgba="0. 0. 0.5 0.2",
                               size="0.025",
                               pos=array_to_string(transformed_point))
            goem_parent_body.append(site)


# -------------------------------------------------------------------------------------------------------------------
#                                                   Classes
# -------------------------------------------------------------------------------------------------------------------
class MogazeHumanEnv(MujocoInitEnv):

    def __init__(self, data_fixed, inv_ind, **kwargs):
        self.data_fixed = data_fixed
        self.inv_ind = inv_ind
        self.goal = None
        super().__init__(**kwargs)
        gen_vhull_sites(self.model)

    # def _load_model(self):
    #     gen_vhull_sites(self.model)

    def set_goal(self, goal: Goal):
        self.goal = goal

    def _get_reference(self):
        # from MJCF to URDF id, keeping MJCF order, reorderring trajectory according to MJCF joints order
        human = self.model
        filter_base_trans_joints = [j for j in human.joints if "basetrans" in j.lower()]
        urdf_fixed_joints = list(self.data_fixed.keys())
        filter_fixed_joints = [j for j in human.joints if j.replace(human.object_id + ":", "") in urdf_fixed_joints]
        filter_other_joints = [j for j in human.joints if not "trans" in j.lower()]
        # ----
        # human mjcf qpos
        self.mjcf_qpos_from_urdf_other_joint_ids = [self.inv_ind[j.replace(human.object_id + ":", "")] for j in
                                                    filter_other_joints]
        self.mjcf_qpos_from_urdf_base_trans_joint_ids = [self.inv_ind[j.replace(human.object_id + ":", "")] for j in
                                                         filter_base_trans_joints]
        self._ref_other_joint_ids = [self.sim.model.joint_name2id(j) for j in filter_other_joints]
        # human mjcf base pos and qpos
        self._ref_base_trans_joint_ids = [self.sim.model.joint_name2id(j) for j in filter_base_trans_joints]

        # human mjcf fixed joints ids (in mjcf) and vals (in trajectory)
        self._ref_fixed_joint_ids = [self.sim.model.joint_name2id(j) for j in filter_fixed_joints]
        self._val_fixed_joints = [self.data_fixed[j.replace(human.object_id + ":", "")] for j in filter_fixed_joints]
        # For CVX Hulls
        self._ref_sites = [self.sim.model.site_name2id(site.get("name")) for site in
                           self.model.worldbody.findall(".//site")]
        # ===============================================
        if self.goal:
            self._ref_goal_body = self.sim.model.body_name2id(self.goal.base_body.get("name"))
        # ===============================================

    def _get_observation_as_dict(self):
        """ Return observations (if any) in the form of OrderedDict({...})"""
        return OrderedDict()

    def step(self, traj):
        self.sim.data.qpos[self._ref_fixed_joint_ids] = self._val_fixed_joints  # traj[len(workplace.sim.data.qpos)]
        if len(traj) == len(self.mjcf_qpos_from_urdf_other_joint_ids):
            self.sim.data.qpos[self._ref_other_joint_ids] = traj[self.mjcf_qpos_from_urdf_other_joint_ids]
        else:
            self.sim.data.qpos[self._ref_base_trans_joint_ids] = traj[self.mjcf_qpos_from_urdf_base_trans_joint_ids]
            self.sim.data.qpos[self._ref_other_joint_ids] = traj[self.mjcf_qpos_from_urdf_other_joint_ids]
        if self.goal:
            self.sim.model.body_pos[self._ref_goal_body] = traj[-3:]

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension. This depends on the model.dof
        """
        low = np.ones(self.model.dof) * -1.
        high = np.ones(self.model.dof) * 1.
        return low, high

    @property
    def joint_qpos(self):
        """ ordered as in original trajectory"""
        traj = np.zeros(len(self._ref_base_trans_joint_ids) + len(self._ref_other_joint_ids))
        traj[self.mjcf_qpos_from_urdf_base_trans_joint_ids] = self.sim.data.qpos[self._ref_base_trans_joint_ids]
        traj[self.mjcf_qpos_from_urdf_other_joint_ids] = self.sim.data.qpos[self._ref_other_joint_ids]
        return traj

    @property
    def base_trans_joints(self):
        """
        Returns base translation joints
        """
        return self.sim.data.qpos[self._ref_base_trans_joint_ids].copy()

    @property
    def non_fixed_other_joints(self):
        """
        Returns base rotation joints + other rotation joints that are NOT FIXED
        """
        return self.sim.data.qpos[self._ref_other_joint_ids].copy()

    @property
    def site_pos(self):
        return self.sim.data.site_xpos[self._ref_sites].copy()


def build_env(has_renderer=True, scenario="p1_1"):
    task = Task().merge_arena(EmptyArena())
    goal = Goal()
    task.merge_objects([goal])

    humans = [EmptyObject("human",
                          fname=os.path.join(os.path.dirname(__file__), "model/assets/human.xml")) for _ in range(13)]
    for human in humans[1:]:
        geoms = human.worldbody.findall(".//geom")
        sites = human.worldbody.findall(".//site")
        for element in geoms + sites:
            element.set("rgba", "0.7 1 .3 0.4")

        for element in geoms:
            element.set("contype", "0")
            element.set("conaffinity", "0")

    full_traj = Trajectory()
    traj_path = os.path.join(os.path.dirname(__file__),
                             "../../data/mogaze_data/{}/{}_human_data.hdf5".format(scenario, scenario))
    full_traj.loadTrajHDF5(traj_path)
    actionables = [MogazeHumanEnv(full_traj.data_fixed, full_traj.inv_ind, model=human) for human in humans]
    actionables[0].set_goal(goal)
    workplace = WorkplaceEnv(task, actionables=actionables, has_renderer=has_renderer)
    # disable collisions
    for actionable in actionables:
        workplace.sim.model.site_rgba[actionable._ref_sites] = 0
    workplace.sim.model.nconmax = 0
    workplace.sim.model.njmax = 0

    return workplace


@click.command()
@click.option("--scenario", "-s", help='select one scenario: p1_1 OR p2_1',
              default="p2_1")
def run(scenario):
    real_data, all_trajs = data_utils.load_mogaze_data(config.MOGAZE_DATA_DIR, actions=[scenario], limit=19500)
    workplace = build_env()
    actionables = workplace.actionables
    n_humans = len(workplace.actionables)
    workplace.set_viewer_camera()
    while True:
        for itraj, traj in enumerate(all_trajs):
            actionables[0].step(traj)
            for actionable, traj_fut in zip(actionables[1:], all_trajs[itraj + 1::5]):
                actionable.step(traj_fut)
            workplace.sim.forward()
            # workplace.viewer.viewer._paused = True
            workplace.render()


if __name__ == '__main__':
    run()
