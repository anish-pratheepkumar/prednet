import warnings

import numpy as np
from gym import error

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    # TODO Add obstacle joints here as well
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """
    For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if action is None:
        return
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def reset_actuators(sim):
    if sim.data.ctrl is not None:
        for i in range(len(sim.data.ctrl)):
            idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
            sim.data.ctrl[i] = sim.data.qpos[idx]


def reset_actuators_with_ids(sim, joint_ids, actuator_ids):
    if sim.data.ctrl is not None:
        sim.data.ctrl[actuator_ids] = sim.data.qpos[joint_ids]
    else:
        warnings.warn("sim.data.ctrl is none. Are you sure you have actuators in your env?")


def ctrl_set_action_with_actuator_ref(sim, actuator_ids, action, is_delta=False):
    """
    For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.data.ctrl is not None:
        # print(action)
        for i, actuator_id in enumerate(actuator_ids[:len(action)]):
            if sim.model.actuator_biastype[actuator_id] == 0:
                sim.data.ctrl[actuator_id] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[actuator_id, 0]]
                sim.data.ctrl[actuator_id] = sim.data.qpos[idx]  # first reset actuator, then control it
                if is_delta:
                    sim.data.ctrl[actuator_id] = sim.data.qpos[idx] + action[i]
                else:
                    sim.data.ctrl[actuator_id] = action[i]


def ctrl_set_action_with_joint_ref(sim, joint_ids, action, is_delta=False):
    if is_delta:
        for a, j_id in zip(action, joint_ids):
            sim.data.qpos[j_id] += a
    else:
        for a, j_id in zip(action, joint_ids):
            sim.data.qpos[j_id] = a
    # sim.forward()


def mocap_set_action(sim, action, mocap_id=0, is_action_delta=True, ignore_pos=False, ignore_quat=False):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    # print(action)
    if sim.model.nmocap > 0:
        # action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(1, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim, mocap_body_id=mocap_id, ignore_quat=ignore_quat)
        if is_action_delta:
            # reset mocap position to robot gripper, then move it
            mocap_pos = sim.data.mocap_pos[mocap_id] + pos_delta
            mocap_quat = sim.data.mocap_quat[mocap_id] + quat_delta
        else:
            mocap_pos = pos_delta
            mocap_quat = quat_delta
        if not ignore_pos:
            sim.data.mocap_pos[mocap_id] = mocap_pos
        if not ignore_quat:
            sim.data.mocap_quat[mocap_id] = mocap_quat


def reset_mocap2body_xpos(sim, mocap_body_id=None, ignore_quat=False):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
            sim.model.eq_obj1id is None or
            sim.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_body_id is not None and mocap_id != mocap_body_id:
            continue
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        if not ignore_quat:
            sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def reset_relpose(sim, body_name, value):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """
    if (sim.model.eq_type is None or
            sim.model.eq_obj1id is None or
            sim.model.eq_obj2id is None):
        return
    for eq_id, (eq_type, obj1_id, obj2_id) in enumerate(zip(sim.model.eq_type,
                                                            sim.model.eq_obj1id,
                                                            sim.model.eq_obj2id)):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            body_idx = obj1_id
        eq_body_name = sim.model.body_id2name(body_idx)

        if body_name is eq_body_name: # todo bug: is should be replace by ==
            sim.model.eq_data[eq_id] = value
