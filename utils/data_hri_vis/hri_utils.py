import os
import sys

import click

project_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, project_dir)
os.environ['PATH'] += os.pathsep + project_dir
from collections import OrderedDict

import numpy as np

import utils.data_generation.data_utils as data_utils
from libs.robosuite.environments.base import MujocoInitEnv
from libs.robosuite.environments.workplaces import WorkplaceEnv
from libs.robosuite.models.arenas import EmptyArena
from libs.robosuite.models.objects.objects import EmptyObject, Goal
from libs.robosuite.models.tasks import Task
# -------------------------------------------------------------------------------------------------------------------
#                                                   Classes
# -------------------------------------------------------------------------------------------------------------------
from libs.robosuite.utils.robot_utils import get_body


class HRIHumanEnv(MujocoInitEnv):

    def __init__(self, **kwargs):
        self.goal = None
        super().__init__(**kwargs)
        # gen_vhull_sites(self.model)

    def set_goal(self, goal: Goal):
        self.goal = goal

    def _get_reference(self):
        # Body ids
        root = get_body(self.model.worldbody, "root").get("name")
        self.root_body_id = self.sim.model.body_name2id(root)
        self.robot_joints = list(self.model.joints)[1:]  # ignore first join?
        self._ref_joint_pos_indexes = np.array([self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints])

        self._ref_root_pos_index = [self.sim.model.get_joint_qpos_addr(root)]
        self._ref_root_pos_index = list(range(self._ref_root_pos_index[0][0], self._ref_root_pos_index[0][1]))

        self._ref_qpos_indices = np.append(self._ref_root_pos_index, self._ref_joint_pos_indexes)

        # ===============================================
        if self.goal:
            self._ref_goal_body = self.sim.model.body_name2id(self.goal.base_body.get("name"))
        # ===============================================
        # For CVX Hulls
        self._ref_sites = [self.sim.model.site_name2id(site.get("name")) for site in
                           self.model.worldbody.findall(".//site")]

    def _get_observation_as_dict(self):
        """ Return observations (if any) in the form of OrderedDict({...})"""
        return OrderedDict()

    @property
    def joint_qpos(self):
        return self.sim.data.qpos[self._ref_qpos_indices]

    @joint_qpos.setter
    def joint_qpos(self, qpos):
        self.sim.data.qpos[self._ref_qpos_indices] = qpos[:len(self._ref_qpos_indices)]  # [35:70]

    def step(self, traj):
        self.joint_qpos = traj
        if self.goal:
            self.sim.model.body_pos[self._ref_goal_body] = traj[-3:]
            # self.sim.data.body_xpos[self._ref_goal_body] = traj[-3:]

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension. This depends on the model.dof
        """
        low = np.ones(self.model.dof) * -1.
        high = np.ones(self.model.dof) * 1.
        return low, high

    @property
    def site_pos(self):
        return self.sim.data.site_xpos[self._ref_sites].copy()


def build_env(has_renderer=True):
    task = Task().merge_arena(EmptyArena())
    goal = Goal()
    task.merge_objects([goal])
    humans = [
        EmptyObject("human", fname=os.path.join(os.path.dirname(__file__), "model/assets/human_deepmimic_vol.xml"),
                    add_base=False)
        for _ in range(13)]
    for human in humans[1:]:
        geoms = human.worldbody.findall(".//geom")
        sites = human.worldbody.findall(".//site")
        for element in geoms + sites:
            element.set("rgba", "0.7 1 .3 0.4")
        for element in geoms:
            element.set("contype", "0")
            element.set("conaffinity", "0")
    actionables = [HRIHumanEnv(model=human) for human in humans]

    actionables[0].set_goal(goal)
    workplace = WorkplaceEnv(task, actionables=actionables, has_renderer=has_renderer)
    # disable collisions
    # hide CVX points
    for actionable in actionables:
        workplace.sim.model.site_rgba[actionable._ref_sites] = 0
    workplace.sim.model.nconmax = 0
    workplace.sim.model.njmax = 0

    return workplace


@click.command()
@click.option("--scenario", "-s", help='select one scenario: co_existing, co_operating, noise OR combined',
              default="combined")
@click.option("--dataset_type", "-t", help='select dataset type: train OR test',
              default="train")
def run(scenario, dataset_type):
    path_to_dataset = os.path.join(os.path.dirname(__file__), "../../data/hri_data/hri_scenarios.h5")
    all_episodes = data_utils.load_hri_data(path_to_dataset=path_to_dataset,
                                            action=scenario,
                                            dataset_type=dataset_type)  # train or test
    print(len(all_episodes))
    workplace = build_env()
    workplace.set_viewer_camera(lookat=(0., 0., 0.))
    actionables = workplace.actionables
    for episode in all_episodes:
        for itraj, traj in enumerate(episode):
            actionables[0].step(traj)
            for actionable, traj_fut in zip(actionables[1:], episode[itraj + 1::5]):
                actionable.step(traj_fut)
            workplace.sim.forward()
            # workplace.viewer.viewer._paused = True
            workplace.render()


if __name__ == '__main__':
    run()
