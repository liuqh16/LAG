import os
import numpy as np
import gym
from gym.spaces import Box, Discrete
from abc import ABC, abstractmethod
from ..core.catalog import Catalog
from ..utils.utils import parse_config, get_root_dir


class BaseTask(ABC):
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    def __init__(self, config):
        # parse config
        self.config = parse_config(os.path.join(get_root_dir(), 'configs', config))
        self.init_variables()
        self.reward_functions = []
        self.termination_conditions = []

    @abstractmethod
    def init_variables(self):
        self.action_var = None
        self.state_var = None

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

    def reset(self, env):
        """Task-specific reset

        Args:
            env: environment instance
        """
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

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
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success or s
            if done:
                break
        return done, info
