import gym
import numpy as np
from ..core.simulation import Simulation
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config


class BaseEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "csv"]}

    def __init__(self, config: str):
        self.config = parse_config(config)
        self.num_agents = getattr(self.config, 'num_agents', 1)
        self.max_steps = getattr(self.config, 'max_steps', 100)
        self.jsbsim_freq = getattr(self.config, 'jsbsim_freq', 60)
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)
        self.load()

    def load(self):
        self.load_task()
        self.load_variables()

    def load_task(self):
        self.task = BaseTask(self.config)
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def load_variables(self):
        self.init_longitude, self.init_latitude = 120.0, 60.0
        self.init_conditions = [None] * self.num_agents
        self.sims = [None] * self.num_agents
        self.current_step = 0

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Args:
            init_conditions (np.array): the initial observation of the space.
        """
        self.current_step = 0
        self.close()

        self.sims = [Simulation(aircraft_name='f16',
                                init_conditions=self.init_conditions[i],
                                origin_point=(self.init_longitude, self.init_latitude),
                                jsbsim_freq=self.jsbsim_freq,
                                agent_interaction_steps=self.agent_interaction_steps) for i in range(self.num_agents)]

        next_observation = self.get_observation()
        self.task.reset(self)
        return next_observation

    def step(self, actions):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and 
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.array): the agents; action, with same length as action variables

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward: amount of reward returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {}

        next_observation = self.make_step(actions)

        reward, info = self.task.get_reward(self, 0, info)
        done, info = self.task.get_termination(self, 0, info)

        return next_observation, reward, done, info

    def make_step(self, actions: list):
        """Calculates new state.

        Args:
            actions (np.array): agents' last action.

        Returns:
            (np.array): agents' observation of the environment state
        """
        # take actions
        for agent_id in range(self.num_agents):
            self.sims[agent_id].set_property_values(self.task.action_var, actions[agent_id])
            self.sims[agent_id].run()

        return self.get_observation()

    def get_observation(self):
        """get state observation from sim.

        Returns:
            (np.array): the first state observation of the episode
        """
        next_observation = []
        for agent_id in range(self.num_agents):
            next_observation.append(self.sims[agent_id].get_property_values(self.task.state_var))
        return np.array(next_observation)

    def get_sim_time(self):
        """ Gets the simulation time from sim, a float. """
        return self.sims[0].get_sim_time()

    def close(self):
        """Cleans up this environment's objects
        Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self.sims:
            if sim:
                sim.close()

    def render(self, mode="human", **kwargs):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - csv: output to cvs files

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return
