from collections import OrderedDict

import numpy as np

from ..base import MujocoInitEnv
from ...utils.mocap_utils import ctrl_set_action_with_actuator_ref, reset_actuators, \
    reset_mocap2body_xpos


class RobotEnv(MujocoInitEnv):
    """Initializes a robot environment or anything which can have a gripper!"""

    def __init__(
            self,
            add_force_torque_sensor=False,
            model=None,
            control_freq=-1,
            dof=None,
            **_
    ):

        self.base_name = None

        self.add_force_torque_sensor = add_force_torque_sensor
        control_freq = control_freq or model.control_freq
        super().__init__(model, control_freq=control_freq)

        if dof is not None:
            self.dof = dof

    def _load_model(self):
        # self.model.save_model("test_{}.xml".format(self.model.object_name))
        pass

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        # print("resetting the env")
        self.stop_flag = False
        if len(self.model.init_qpos) <= len(self._ref_joint_pos_indexes):
            self.sim.data.qpos[self._ref_joint_pos_indexes[:len(self.model.init_qpos)]] = self.model.init_qpos
        else:
            self.sim.data.qpos[self._ref_joint_pos_indexes] = self.model.init_qpos[:len(self._ref_joint_pos_indexes)]
        # self.sim.data.qpos[self._ref_joint_pos_indexes[:len(self.model.init_qpos)]] = [0.68241,-0.70706,2.30274,0.,1.57,-1.57]
        self.sim.forward()
        # get end effector orientation to set the gripper rotation

        # grip_pose = self.grip_pose

        if self.model.control_using_mocap_not_joints:
            # Instead of resetting mocap --> reset relpose
            reset_mocap2body_xpos(self.sim, ignore_quat=False)
            # mocap_orientation = self.sim.data.mocap_quat[self.mocap_id]

        elif self.model.control:
            reset_actuators(self.sim)

        self.sim.forward()

        """
        mocap_set_action(self.sim, np.concatenate([gripper_target, gripper_rotation]), is_action_delta=False,
                         ignore_quat=False)
        for _ in range(10):
            self.sim.step()
        """

    def get_base_pos(self):
        return self.sim.data.body_xpos[self._base_body_id]  # , self.sim.data.body_qpos[body_id]

    def set_base_pos(self, pos):
        self.sim.model.body_pos[self._base_body_id][:len(pos)] = pos
        self.sim.forward()

    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        self._ref_joint_actuator_indexes = []
        # base body name and id
        base_body = self.model.base_body
        self.base_name = base_body.get("name")
        self._base_body_id = self.sim.model.body_name2id(self.base_name)

        self.all_bodies_ids = set([self.sim.model.body_name2id(body_name) for body_name in self.model.all_bodies_names])

        # Robot joints names and ids
        self.robot_joints = list(self.model.joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_pos_indexes = []
        for x in self.robot_joints:
            joint_range = self.sim.model.get_joint_qpos_addr(x)
            if isinstance(joint_range, tuple):
                self._ref_joint_pos_indexes.extend(list(range(*joint_range)))
            else:
                self._ref_joint_pos_indexes.append(joint_range)

    def _pre_action(self, action):
        if action is None:
            return  # Do nothing!
        self._set_action(action)

    @property
    def joint_qpos(self):
        return self.sim.data.qpos[self._ref_joint_pos_indexes].copy()

    @joint_qpos.setter
    def joint_qpos(self, qpos):
        self.sim.data.qpos[self._ref_joint_pos_indexes] = qpos

    @property
    def joint_qvel(self):
        """Returns the joint velocities."""
        return self.sim.data.qvel[self._ref_joint_pos_indexes]

    @joint_qvel.setter
    def joint_qvel(self, qvel):
        """Returns the joint velocities."""
        self.sim.data.qvel[self._ref_joint_pos_indexes] = qvel

    def mocap_pos(self):
        return self.sim.data.body_xpos[self.mocap_body_id].copy()

    def _set_action(self, action):
        if self.model.control:  # control using joints
            assert len(action) == len(self._ref_joint_actuator_indexes)
            ctrl_set_action_with_actuator_ref(self.sim, self._ref_joint_actuator_indexes, action,
                                              self.is_robot_action_delta)

    def _post_action(self, *_, **__):
        """
        (Optional) does gripper visualization after actions.
        """
        return None, None, {}

    def _get_observation_as_dict(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """
        di = OrderedDict()
        return di

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers already included in the model).
        """
        return self.model.dof

    @dof.setter
    def dof(self, dof):
        """
        Returns the DoF of the robot (with grippers already included in the model).
        """
        self.model.dof = dof
