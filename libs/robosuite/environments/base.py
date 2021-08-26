import abc
import os
import time
import warnings
from collections import Iterable
from collections import OrderedDict
from math import ceil

import gym
import numpy as np
from gym import spaces
from mujoco_py import MjSim, MjRenderContextOffscreen, MujocoException
from mujoco_py import load_model_from_xml

from ..models.mjcf_utils import array_to_string
from ..models.object_base import ObjectBase
from ..utils import SimulationError, XMLError
from ..utils.mujoco_py_renderer import MujocoPyRenderer
from ..utils.robot_utils import get_body
from ..utils.transform_utils import quat_to_euler

REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments. When an environment is created, it is registered by defaults"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "SawyerEnv", "BaxterEnv", "Ur5Env"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoInitEnv(abc.ABC):
    """
    Mujoco Env initializer. Has to be called in extending classes using MujocoInitEnv.__init__(self)
    TODO Any object which needs to load an xml file, steps, support/exclude collisions or get_reference to be referenced in
    the workspace
    """

    def __init__(self, model: ObjectBase = None, control_freq=-1):
        """

        Args:
            model:
            control_freq: if not set, it will take default control_freq of the WorkplaceEnv
        """
        self.sim = None
        self.env = None
        self.model = model
        self.control_freq = control_freq

        self.update_interval_ = None
        self.dt_ = None  # time in seconds for each robot control: self.update_interval*self.sim.timesteps*self.sim.nsubsteps

    def init(self, sim, env):
        self.sim = sim
        self.env = env

    @property
    def control_freq(self):
        return self.control_freq_

    @control_freq.setter
    def control_freq(self, control_freq):
        self.response_time = 1 / control_freq
        self.control_freq_ = control_freq

    @property
    def dt(self):
        """

        Returns: the timestep size according to the simulation settings

        """
        if self.dt_:
            return self.dt_
        else:
            if self.sim is None:
                return 0.002  # default sim.opt.timestep
            else:
                self.dt_ = self.sim.model.opt.timestep * self.sim.nsubsteps
            return self.dt_

    @property
    def update_interval(self):
        """
        the time interval to call .step() on this env. The time interval depends on the simulated time.
        We simulate self.sim.model.opt.timestep * n_substeps (in seconds, default is 0.002*1). So if our robot control frequence is
        125 Hz (like Ur5), then we can send Ur5 an action every update_interval (1/125 /(0.002*n_substeps)
        Returns: X [simulation_timesteps]. Integer multiple of the simulation_timesteps for one update.
        One simulation timestep values to 0.002 seconds, so X can be converted from X [timesteps] to X*0.002 [s]. Minimum timestep size is 1

        """
        if self.update_interval_ is None:
            if self.sim is None:
                return 0.  # 0.04
            else:
                # multiply simulation time in seconds by control interval
                if self.control_freq <= -1:
                    self.control_freq = 1 / self.sim.model.opt.timestep * self.sim.nsubsteps
                self.update_interval_ = int(ceil(
                    (1. / self.control_freq) / (self.sim.model.opt.timestep * self.sim.nsubsteps)))
        return self.update_interval_  # default update rate is set to 1 timestep if the control_freq is not set

    def _load_model(self):
        """
        (Optional) Do here any modifications to the model before initializing the sim, e.g. changing geom/site
        visualization groups or excluding collisions between obstacle and table
        """
        self.update_interval_ = None  # force recalculate update_interval if simulation changes
        pass

    def _reset_internal(self):
        """
        resets the environment
        """
        pass

    def _get_reference(self):
        """
        Called after sim is initialized. Make the necessary bindings from self.model (the XML) to sim
        """
        raise NotImplementedError()

    def _get_observation_as_dict(self):
        """ Return observations (if any) in the form of OrderedDict({...})"""
        raise NotImplementedError()

    def step(self, action=None):
        """
        Called each timestep (before _set_action is called if self is `Actionable`). Can be used for stepping
        the objects, e.g. moving obstacles, moving goals
        """
        pass

    def render(self):
        self.env.render()

    def __getstate__(self):
        """
        For pickling, ignore any Cython based variables, e.g., sim and env
        Returns:
        """
        state = self.__dict__.copy()
        remove_keys = ('sim', 'env')
        for k in remove_keys:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        remove_keys = ('sim', 'env')
        for k in remove_keys:
            state[k] = None
        self.__dict__.update(state)

    @staticmethod
    def header():
        return ['Type', "Name", "ID", "Base position [m]", "Base orientation [degrees]"]

    def to_dict(self):
        header = MujocoInitEnv.header()
        if self.sim:
            base_body_id = self.sim.model.body_name2id(self.model.base_body.get("name"))
            base_body_pos = array_to_string(self.sim.data.body_xquat[base_body_id])
            base_body_euler = array_to_string(quat_to_euler(self.sim.data.body_xquat[base_body_id]))
        else:
            base_body_pos = self.model.base_body.get("pos", "0. 0. 0.")
            base_body_euler = self.model.base_body.get("euler", "0. 0. 0.")
        return OrderedDict({
            header[0]: type(self).__name__,
            header[1]: self.model.object_name,
            header[2]: self.model.object_id,
            header[3]: base_body_pos,
            header[4]: base_body_euler,
        })


class Observable(abc.ABC):
    def __init__(self, use_her: bool = False):
        self.use_her = use_her
        obs = self._get_observation()
        obs_dict = self._get_observation_as_dict()
        self.decode_obs_keys = self.decode_obs_init(obs_dict)  # for decoding the observation

        if self.use_her:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32')
            ))
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def _get_observation(self):
        raise NotImplementedError()

    def _get_observation_as_dict(self):
        raise NotImplementedError()

    def encode_obs(self, observation) -> np.array:
        if (observation != {}):
            encoded_observation = np.concatenate([value.reshape(-1) for value in observation.values()]).ravel()
        else:
            encoded_observation = np.empty_like(observation, dtype=float)
        return encoded_observation

    def decode_obs_init(self, observation) -> list:
        ds_keys = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                ds_keys.append((key, value.shape))
            else:
                ds_keys.append((key, len(value)))
        return ds_keys


class ActionEnv(abc.ABC):
    def __init__(self, action_space):
        self.action_space = action_space  # TODO should be specified as property of WorkplaceEnv

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        raise NotImplementedError()

    def _pre_action(self, action):
        """Do any preprocessing before taking an action."""
        raise NotImplementedError()

    def _post_action(self, obs, goal, achieved_goal, info={}):
        """Do any housekeeping after taking an action."""
        raise NotImplementedError()


class MujocoEnv(gym.GoalEnv, MujocoInitEnv, ActionEnv):  # TODO add other interfaces
    """Initializes a Mujoco Environment."""

    def __init__(
            self,
            has_renderer=False,
            has_offscreen_renderer=False,
            render_collision_mesh=True,
            render_visual_mesh=True,
            control_freq=100,
            render_freq=60,
            horizon=1000,
            ignore_done=False,
            use_camera_obs=False,
            camera_name="frontview",
            camera_height=256,
            camera_width=256,
            camera_depth=False,
            use_her=False,
            n_substeps=20,
            # action_dof=3,
            nconmax=200,
            njmax=200,
            init_mujoco_py=True,
            **kwargs
    ):
        """
        Args:

            has_renderer (bool): If true, render the simulation state in 
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes 
                in camera. False otherwise.

            control_freq (float): how many control signals to receive 
                in every simulated second. This sets the amount of simulation time 
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a 
                rendered image.

            camera_name (str): name of camera to be rendered. Must be 
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """
        MujocoInitEnv.__init__(self)
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.control_freq = control_freq
        self.render_freq = render_freq
        self.horizon = horizon

        self._max_episode_steps = horizon  # Openai's horizon!

        self.ignore_done = ignore_done
        self.viewer = None
        self.model = None
        self.collision_geoms = None
        self.use_her = use_her

        # settings for camera observations
        self.use_camera_obs = use_camera_obs
        if self.use_camera_obs and not self.has_offscreen_renderer:
            raise ValueError("Camera observations require an offscreen renderer.")
        self.camera_name = camera_name
        if self.use_camera_obs and self.camera_name is None:
            raise ValueError("Must specify camera name when using camera obs")
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_depth = camera_depth
        self.is_depth = False
        self.n_substeps = n_substeps
        # ----------------------------------------------------------------------------------------------------
        #                               Contact Properties for sim and rendering
        # ----------------------------------------------------------------------------------------------------
        self.nconmax = nconmax
        self.njmax = njmax
        if init_mujoco_py:
            self.init_mujoco_py()

            self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0 / self.dt))
            }
            # # TODO action_space should be set automatically from Actionables
            # self.action_space = spaces.Box(-1, 1, shape=(action_dof,),
            #                                dtype='float32')  # TODO should be specified as property of WorkplaceEnv

    def init_mujoco_py(self):
        """
        Loads all models and start a mujoco_py Sim and Viewer
        Returns:

        """
        ObjectBase.object_ids_dict = {}
        start_time = time.time()
        self.close()  # close first any existing environments
        for i in range(10):
            try:
                self.init_sim()
                self._get_reference()
                self._reset_internal()
                break
            except MujocoException as e:
                self.close()
                if "nconmax" in e.__str__():
                    self.nconmax *= 2
                    warnings.warn(
                        "Could not initialize the sim with setup nconmax {}. Increasing both by nconmax".format(
                            self.nconmax))
                if "njmax" in e.__str__():
                    self.njmax *= 2
                    warnings.warn(
                        "Could not initialize the sim with setup njmax {}. Increasing njmax by twice".format(
                            self.njmax))
        self.init_viewer()

        # update observation space
        obs = self._get_observation()
        obs_dict = self._get_observation_as_dict()
        self.decode_obs_keys = self.decode_obs_init(obs_dict)  # for decoding the observation

        if self.use_her:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32')
            ))
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

        total_time = str((time.time() - start_time) / 60) + " min"
        print('Init mujoco_py in {}'.format(total_time))
        return self

    def pause_renderer(self):
        if self.has_renderer:
            self.render()
            self.viewer.viewer._paused = True
            self.render()

    @property
    def dt(self):
        """

        Returns: simulation timestep size in sec

        """
        if self.sim is None:
            # take it from the task
            timestep_in_sec = float(self.task.option.get("timestep", "0.04")) * self.n_substeps
            return timestep_in_sec
        else:
            return self.sim.model.opt.timestep * self.sim.nsubsteps

    def decode_obs_init(self, di):
        ds_keys = []
        for key, value in di.items():
            if isinstance(value, np.ndarray):
                ds_keys.append((key, value.shape))
            elif isinstance(value, Iterable):
                ds_keys.append((key, len(value)))
            else:
                ds_keys.append((key, 1))

        return ds_keys

    def decode_obs(self, observation):
        """
        From observation dict, save each key and the dimension of its value.
        E.g., di_decoded = self.decode_obs(obs[Keys.Obs.OBSERVATION])

        :param observation: dict (str,ndarray)
        :return:
        """
        decoded_di = {}
        i = 0
        for key, dim in self.decode_obs_keys:
            if len(dim) > 1:
                i_end = i + np.prod(np.array(dim))
                decoded_di[key] = np.array(observation[i:i_end]).reshape(dim)
            else:
                i_end = i + dim[0]
                decoded_di[key] = observation[i:i_end]
            i = i_end
        return decoded_di

    def get_indices_of_keys_in_obs(self):
        decoded_di_indices = {}
        i = 0
        for key, dim in self.decode_obs_keys:
            if len(dim) > 1:
                i_end = i + np.prod(np.array(dim))
                decoded_di_indices[key] = (i, i_end)
            else:
                i_end = i + dim[0]
                decoded_di_indices[key] = (i, i_end)
            i = i_end
        return decoded_di_indices

    def encode_obs(self, observation):
        if (observation != {}):
            encoded_di = np.concatenate([value.reshape(-1) for value in observation.values()]).ravel()
        else:
            encoded_di = np.empty_like(observation, dtype=float)
        return encoded_di

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.
        """
        self.cur_time = 0
        self.model_timestep = self.sim.model.opt.timestep
        if self.model_timestep <= 0:
            raise XMLError("xml model defined non-positive time step")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError(
                "control frequency {} is invalid".format(control_freq)
            )
        self.control_timestep = 1. / control_freq

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def reset(self):
        """Resets simulation."""
        # if there is an active viewer window, destroy it
        # self._destroy_viewer()
        if self.sim is not None:
            self._reset_internal()
            self.sim.forward()
            return self._get_observation()
        else:
            return OrderedDict()

    def init_sim(self):
        # instantiate simulation from MJCF model
        self._load_model()
        model_size = self.model.root.find("size")
        # Check if file assets exists, if they do not exist, then try to generate them
        file_assets = self.model.asset.findall(".//mesh")
        for f in file_assets:
            f_path = f.get("file")
            if not os.path.exists(f_path):
                path = os.path.normpath(f_path)
                i = path.rfind("robosuite\\robosuite\\")
                i_length = len("robosuite\\robosuite\\")
                if i == -1:
                    i = path.rfind("robosuite/robosuite/")
                    i_length = len("robosuite/robosuite/")
                if i == -1:
                    continue  # ignore file
                path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), path[i + i_length:])
                f.set("file", path)
            else:
                pass
        model_size.set("nconmax", str(self.nconmax))
        model_size.set("njmax", str(self.njmax))
        self.mjpy_model = self.model.get_model(mode="mujoco_py")
        self.sim = MjSim(self.mjpy_model, nsubsteps=self.n_substeps)
        self.initialize_time(self.control_freq)
        # create visualization screen or renderer

    def init_viewer(self):
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True
            self.viewer._render_every_frame = True
        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (
                1 if self.render_collision_mesh else 0
            )
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (
                1 if self.render_visual_mesh else 0
            )

    def _reset_internal(self):
        """Resets simulation internal configurations."""
        # additional housekeeping
        if self.sim is not None:
            self.sim_state_initial = self.sim.get_state()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

    def _get_observation(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def _get_observation_as_dict(self):
        """Returns an OrderedDict containing observations [(name_string, np.array), ...]."""
        return OrderedDict()

    def step(self, action):
        """Takes a step in simulation with control command @action."""
        # if self.done:
        #    raise ValueError("executing action in terminated episode")

        self.timestep += 1
        self._pre_action(action)
        # end_time = self.cur_time + self.control_timestep
        # while self.cur_time < end_time:
        for _ in range(1):
            self.sim.step()  # TODO wrong, does not consider n_substeps
        self.cur_time += self.model_timestep
        if self.use_her:
            goal = self._get_observation()['desired_goal']
            achieved_goal = self._get_observation()['achieved_goal']
            obs = self._get_observation()  # ['observation']
            reward, done, info = self._post_action(obs, goal, achieved_goal)
        else:
            reward, done, info = self._post_action(np.zeros(1), np.zeros(1), np.zeros(1))
        return self._get_observation(), reward, done, info

    def _pre_action(self, action):
        """Do any preprocessing before taking an action.
        TODO define pre-action in robot itself"""
        self.sim.data.ctrl[:] = action

    def _post_action(self, obs, goal, achieved_goal, info={}):
        """Do any housekeeping after taking an action."""
        reward = self.compute_reward(obs, goal, achieved_goal, info)
        # done if number of elapsed timesteps is greater than horizon
        self.done = self.timestep >= self.horizon and not self.ignore_done  # ) or self.is_success(obs, goal, achieved_goal)
        return reward, self.done, info

    def compute_reward(self, observations, achieved_goal, desired_goal, info):
        return 0

    def is_success(self, observation, achieved_goal, desired_goal):
        return 0

    def render(self, mode="human"):
        """
        Renders to an on-screen window.
        """
        # time.sleep(1 / self.render_freq)
        if self.viewer:
            #    self.viewer.render()
            if mode == 'rgb_array':
                return None
                # self.viewer.render()
                # # window size used for old mujoco-py:
                # width, height = 500, 500
                # # data = self.viewer.viewer.read_pixels(width, height, depth=False)
                # data = self.viewer.viewer._read_pixels_as_in_window()
                # # original image is upside-down, so flip it
                # return data  # data[::-1, :, :]
            else:
                self.viewer.render()

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.
        """
        observation = self._get_observation()
        return observation

    @property
    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    def reset_from_xml_string(self, xml_string):
        """Reloads the environment from an XML description of the environment."""

        # if there is an active viewer window, destroy it
        self.close()

        # load model from xml
        self.mjpy_model = load_model_from_xml(xml_string)
        self.sim = MjSim(self.mjpy_model, nsubsteps=self.n_substeps)
        self.initialize_time(self.control_freq)
        if self.viewer is None and self.has_renderer:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            self.viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

        elif self.has_offscreen_renderer:
            render_context = MjRenderContextOffscreen(self.sim)
            render_context.vopt.geomgroup[0] = 1 if self.render_collision_mesh else 0
            render_context.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0
            self.sim.add_render_context(render_context)

        self.sim_state_initial = self.sim.get_state()
        self._get_reference()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # necessary to refresh MjData
        self.sim.forward()

    def find_contacts(self, geoms_1, geoms_2):
        """
        Finds contact between two geom groups.

        Args:
            geoms_1: a list of geom names (string)
            geoms_2: another list of geom names (string)

        Returns:
            iterator of all contacts between @geoms_1 and @geoms_2
        """
        for contact in self.sim.data.contact[0: self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                yield contact

    def _destroy_viewer(self):
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()
        pass

    def get_viewer_camera(self):
        if not self.viewer or not self.sim:
            return {}
        else:
            return {
                "ref_body_name": None,
                "distance": self.viewer.viewer.cam.distance,
                "azimuth": self.viewer.viewer.cam.azimuth,
                "elevation": self.viewer.viewer.cam.elevation,
                "lookat": self.viewer.viewer.cam.lookat[:].tolist()
            }

    def set_viewer_camera(self, ref_body_name=None, distance=10, azimuth=130, elevation=-35, lookat=None):
        if not self.viewer or not self.sim:
            return
        if lookat is not None:
            self.viewer.viewer.cam.lookat[:] = lookat[:]
        elif ref_body_name:
            body_id = self.sim.model.body_name2id(get_body(self.model.worldbody, ref_body_name).get("name"))
            lookat = self.sim.data.body_xpos[body_id]
            for idx, value in enumerate(lookat):
                self.viewer.viewer.cam.lookat[idx] = value
        self.viewer.viewer.cam.distance = distance
        self.viewer.viewer.cam.azimuth = azimuth  # 180.
        self.viewer.viewer.cam.elevation = elevation  # -90. # -20

    def __getstate__(self):
        """
        For pickling, ignore any Cython based variables, e.g., sim and env
        Returns:
        """
        state = self.__dict__.copy()
        remove_keys = ('sim', 'env', 'viewer', 'mjpy_model', 'sim_state_initial',)
        for k in remove_keys:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        remove_keys = ('sim', 'env', 'viewer', 'mjpy_model', 'sim_state_initial',)
        for k in remove_keys:
            state[k] = None
        self.__dict__.update(state)
        # self.init_mujoco_py()  # start the environment
