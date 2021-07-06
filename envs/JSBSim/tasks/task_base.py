from types import MethodType
import numpy as np
import gym
from gym.spaces import Box, Discrete
from ..core.catalog import Catalog


class Task:
    """
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and agent_reward function.
    """

    def __init__(self):
        self.action_var = None
        self.state_var = None
        self.init_condition = None
        self.output = None
        self.jsbsim_freq = 60
        self.agent_interaction_steps = 5
        self.aircraft_name = "A320"

        # set default output to state_var
        if self.output is None:
            self.output = self.state_var

        # modify Catalog to have only the current task properties
        names_away = []
        for name, prop in Catalog.items():
            if not (
                prop in self.action_var
                or prop in self.state_var
                or prop in self.init_condition
                or prop in self.output
            ):
                names_away.append(name)
        for name in names_away:
            Catalog.pop(name)

    def get_reward(self, state, sim):
        return 0

    def is_terminal(self, state, sim):
        return False

    def get_observation_var(self):
        return self.state_var

    def get_action_var(self):
        return self.action_var

    def get_initial_conditions(self):
        return self.init_condition

    def get_output(self):
        return self.output

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

    def render(self, sim, mode="human", **kwargs):
        pass

    def define_aircraft(self, aircraft_name="A320"):
        self.aircraft_name = aircraft_name

    def define_state(self, states=None):
        self.state_var = states

    def define_action(self, actions=None):
        self.action_var = actions

    def define_init_conditions(self, init_condition=None):
        self.init_condition = init_condition

    def define_output(self, output=None):
        self.output = output

    def define_jsbsim_freq(self, freq=60):
        self.jsbsim_freq = freq

    def define_agent_interaction_steps(self, steps=5):
        self.agent_interaction_steps = steps

    def define_reward(self, func):
        self.get_reward = MethodType(func, self)

    def define_is_terminal(self, func):
        self.is_terminal = MethodType(func, self)