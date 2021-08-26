import logging
from collections import OrderedDict

import numpy as np
from Cython.Utils import OrderedSet
from gym import spaces
from gym.utils import seeding

from .. import MujocoEnv
from ...models.tasks import Task
from ...models.workplaces.workplace import WorkplaceModel

log = logging.getLogger(__file__)


class WorkplaceEnv(MujocoEnv):
    # TODO instead of passing robot and task, pass any MujocoInitEnv which has step, load_preference, ...
    #  and just loop on them
    def __init__(self,
                 task: Task,
                 actionables=(),
                 observables=(),
                 n_substeps: int = 1,
                 has_renderer=False,
                 use_her=False,
                 horizon=np.Inf,
                 sub_workplaces=None,
                 **kwargs):
        self.task = task
        # ------- Initialize Robot and goal models------

        # Robot and its properties
        self.actionables = actionables or []
        self.observables = observables or []

        self.unique_actionables_observable = []

        self.task = task

        # control using sub_workplaces
        self.sub_workplaces = sub_workplaces or []
        if len(self.sub_workplaces) == 1:
            use_her = self.sub_workplaces[0].use_her
            self.use_openai_her = self.sub_workplaces[0].use_openai_her

        super().__init__(
            horizon=horizon,  # endless time horizon
            n_substeps=n_substeps,
            use_her=use_her,
            has_renderer=has_renderer,
            **kwargs
        )

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed=seed)

    def _load_model(self):
        models = []
        # self.unique_actionables_observable = list(
        #     OrderedDict.fromkeys(list(set(self.observables)) + list(set(self.actionables))).keys())
        unique_actionables_observable_set = OrderedSet(self.observables + self.actionables)
        if len(self.sub_workplaces) > 0:
            unique_observables = OrderedSet()
            unique_actionables = OrderedSet()
            for sub_workplace in self.sub_workplaces:
                unique_observables.update(sub_workplace.observables)
                unique_actionables.update(sub_workplace.actionables)
            unique_actionables_observable_set.update(unique_observables)
            unique_actionables_observable_set.update(unique_actionables)

        self.unique_actionables_observable = list(unique_actionables_observable_set)

        for something in self.unique_actionables_observable:
            # TODO check if model is already loaded, do not load it again
            something._load_model()
            if hasattr(something.model, 'merged_already') or something.model is None:
                continue
            else:
                models.append(something.model)
        workplace_model = WorkplaceModel(robots=models, task=self.task)
        self.model = workplace_model

    def init_sim(self):
        super().init_sim()
        for something in self.unique_actionables_observable:
            something.init(sim=self.sim, env=self)
        for sub_workplace in self.sub_workplaces:
            sub_workplace.init(sim=self.sim, env=self)

    def _get_reference(self):
        super()._get_reference()
        for something in self.unique_actionables_observable:
            something._get_reference()
        low, high = self.action_spec
        if len(low) > 0:
            self.action_space = spaces.Box(np.array(low), np.array(high), dtype='float32')
        else:
            self.action_space = None
        #   TODO       # self.observation_space = spaces.Box(-np.Inf, np.Inf, shape=(len(obs),), dtype='float32')

    def _reset_internal(self):
        for something in self.unique_actionables_observable:
            something._reset_internal()
        super()._reset_internal()

    @property
    def action_spec(self, dim=7, low=-1, high=1):
        # For reach env, output is only x,y,z
        # dof = []
        if len(self.actionables) == 1:
            try:
                return self.actionables[0].action_spec
            except Exception as e:
                log.warning('Actionable {} does not have action_spec'.format(self.actionables[0].model.object_name),
                            exc_info=True)
                return [], []
        else:
            low = []
            high = []
            for something in self.actionables:
                try:
                    action_low, action_high = something.action_spec
                    low.extend(action_low)
                    high.extend(action_high)
                except Exception as e:
                    log.warning('Actionable {} does not have action_spec'.format(something.model.object_name),
                                exc_info=True)
                    continue
        return low, high

    def _pre_action(self, action):
        if action is not None:
            if isinstance(action, (np.ndarray, np.generic)):
                action = action.reshape((len(self.actionables), -1))
                for something, action_ in zip(self.actionables, action):
                    if int(self.timestep % something.update_interval) == 0:
                        something._pre_action(action_)
            elif isinstance(action, dict):
                for something, action_ in action.items():
                    if int(
                            self.timestep % something.update_interval) == 0:  # TODO does this make sense? Because the action if not executed will be lost! Needs to be handled by a scheduler
                        something._pre_action(action_)
            else:
                for something, action_ in zip(self.actionables, action):
                    if int(self.timestep % something.update_interval) == 0:
                        something._pre_action(action_)

    def _post_action(self, obs, goal, achieved_goal, info=None):
        """
        (Optional) does gripper visualization after actions. actionable.update_interval (i.e. control frequency)
         is not applied in post_action
        """
        if info is None:
            info = {}
        for something in self.actionables:
            something._post_action(obs, goal, achieved_goal, info)
        reward, done, info = super()._post_action(obs, goal, achieved_goal, info=info)
        return reward, done, info

    def render(self, mode=None):
        return super().render(mode)

    def _get_observation_as_dict(self):
        """

        :return: observations as dict so that you can add more elements to the dict
        """
        di = OrderedDict()
        for something in self.observables:
            observation = something._get_observation_as_dict()
            if observation is not None:
                di.update(observation)
        return di

    def _get_observation(self):
        """
        TODO flatten ordered dict, and deflatten ordereddict
        DO NOT extend this method. Instead, extend #_get_observation_as_dict(self)
        :return: ravel the dict and return an array or in case of her return a dict
        """
        di_obs = self._get_observation_as_dict()
        di = self.encode_obs(di_obs)
        return di

    def step(self, action):
        # ---------------------------------------------------------------------------------------------
        #                                    Step all observables
        # ---------------------------------------------------------------------------------------------
        for observable in self.observables:
            observable.step(action)
        # ---------------------------------------------------------------------------------------------
        #                                    Control using SubWorkplace
        # ---------------------------------------------------------------------------------------------
        if len(self.sub_workplaces) > 0:
            step_result = {}
            if len(self.sub_workplaces) == 1:
                obs, reward, done, info = self.sub_workplaces[0].step(action)
                _, _, done_horizon, _ = super().step(None)  # to update simulation timesteps
                return obs, reward, done or done_horizon, info
            elif isinstance(action, dict):
                for sub_workplace, action_ in action.items():
                    sub_workplace.step(action_)  # change sim
                    # obs, reward, done, info = sub_workplace.step(action_)

                self.timestep += 1
                self.sim.step()
                self.cur_time += self.model_timestep

                for sub_workplace in self.sub_workplaces:
                    reward, done, info = self._post_action(np.zeros(1), np.zeros(1), np.zeros(1), {})
                    obs = sub_workplace._get_observation()
                    step_result[sub_workplace] = obs, reward, done, info
                return step_result
            else:
                raise NotImplementedError("Only dict of {subworkplace1: array[actions_for_subworkplace1]} is supported")
            # super().step(None)  # to update simulation timesteps
            # return step_result
        else:
            # ---------------------------------------------------------------------------------------------
            #                            Control using Actionables and Observables
            # ---------------------------------------------------------------------------------------------
            # TODO should be : for action in actions: action.resource.step(action)
            for something in self.unique_actionables_observable:
                # current_step = self.timestep  # int(self.sim.data.time / self.dt)
                # support control_frequency for each actionable / observable
                if int(self.timestep % something.update_interval) == 0:
                    something.step(action)
            return super().step(action)

    def close(self):
        for something in self.unique_actionables_observable:
            # TODO check if model is already loaded, do not load it again
            if hasattr(something, "close"):
                something.close()  # mainly for cleanly closing ros_envs if any
        super().close()
