import os
import numpy as np
from gym import spaces
from abc import ABC, abstractmethod
from ..core.catalog import Catalog as c

class BaseTask(ABC):
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    def __init__(self, config):
        self.config = config
        self.reward_functions = []
        self.termination_conditions = []
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    @abstractmethod
    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_ft,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]

    @abstractmethod
    def load_observation_space(self):
        """
        Load observation space
        """
        space_tuple = ()
        for prop in self.state_var:
            if prop.spaces is spaces.Box:
                space_tuple += (spaces.Box(low=np.array([prop.min]), high=np.array([prop.max]), dtype="float"),)
            elif prop.spaces is spaces.Discrete:
                space_tuple += (spaces.Discrete(prop.max - prop.min + 1),)
        self.observation_space = spaces.Tuple(space_tuple)

    @abstractmethod
    def load_action_space(self):
        """
        Load action space
        """
        space_tuple = ()
        for prop in self.action_var:
            if prop.spaces is spaces.Box:
                space_tuple += (spaces.Box(low=np.array([prop.min]), high=np.array([prop.max]), dtype="float"),)
            elif prop.spaces is spaces.Discrete:
                space_tuple += (spaces.Discrete(prop.max - prop.min + 1),)
        self.action_space = spaces.Tuple(space_tuple)

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
