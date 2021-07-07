import os
import numpy as np
import gym
from gym.spaces import Box, Discrete
from abc import abstractmethod, ABCMeta
from ..core.catalog import Catalog
from ..utils.utils import parse_config, get_root_dir


class BaseTask:
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    __metaclass__ = ABCMeta
    def __init__(self, config: str):
        # parse config
        self.config = parse_config(os.path.join(get_root_dir(), 'configs', config))

        self.init_variables()
        self.init_conditions()

        # modify Catalog to have only the current task properties
        names_away = []
        for name, prop in Catalog.items():
            if not (
                prop in self.action_var
                or prop in self.state_var
            ):
                names_away.append(name)
        for name in names_away:
            Catalog.pop(name)

        # set controlling frequency
        self.jsbsim_freq = self.config.jsbsim_freq
        self.agent_interaction_steps = self.config.agent_interaction_steps

        self.reward_functions = []
        self.termination_conditions = []

    @abstractmethod
    def init_variables(self):
        self.action_var = None
        self.state_var = None

    @abstractmethod
    def init_conditions(self):
        self.aircraft_name = None
        self.init_condition = None

    @abstractmethod
    def get_observation_space(self):
        """
        Get the task's observation Space object
        :return : spaces.Tuple composed by spaces of each property.
        """
        space_tuple = ()

        for prop in self.state_var:
            if prop.spaces is Box:
                space_tuple += (Box(low=np.array([prop.min]), high=np.array([prop.max]), dtype="float"),)
            elif prop.spaces is Discrete:
                space_tuple += (Discrete(prop.max - prop.min + 1),)
        return gym.spaces.Tuple(space_tuple)

    @abstractmethod
    def get_action_space(self):
        """
        Get the task's action Space object
        :return : spaces.Tuple composed by spaces of each property.
        """
        space_tuple = ()

        for prop in self.action_var:
            if prop.spaces is Box:
                space_tuple += (Box(low=np.array([prop.min]), high=np.array([prop.max]), dtype="float"),)
            elif prop.spaces is Discrete:
                space_tuple += (Discrete(prop.max - prop.min + 1),)
        return gym.spaces.Tuple(space_tuple)

    @abstractmethod
    def reset_task(self):
        """Task-specific reset
        """
        raise NotImplementedError

    def get_reward(self, env, agent_id, info={}):
        """
        Aggregate reward functions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                reward(float): total reward of the current timestep
                info(dict): additional info
        """
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env, agent_id)
        return reward, info

    def get_termination(self, env, agent_id, info={}):
        """
        Aggregate termination conditions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                done(bool): whether the episode has terminated
                info(dict): additional info
        """
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env, agent_id)
            done = done or d
            success = success or s
        return done, info

    @abstractmethod
    def render(self, sim, mode="human", **kwargs):
        raise NotImplementedError
