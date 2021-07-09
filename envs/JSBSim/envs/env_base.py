import gym
import numpy as np
from ..core.simulation import Simulation
from ..tasks.task_base import BaseTask


class BaseEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "csv"]}

    def __init__(self, config):
        self.sim = None
        self.task = BaseTask()
        self.observation_space = self.task.get_observation_space()
        self.action_space = self.task.get_action_space()

        self.current_step = 0
        self.state = None
        self.action = None

    def step(self, action=None):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and 
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.array, optional): the agent's action, with same length as action variables. Defaults to None.

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward_visualize: amount of reward_visualize returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        self.action = action
        if action is not None:
            if not len(action) == len(self.action_space.spaces):
                raise ValueError("mismatch between action and action space size")

        self.state = self.make_step(action)

        reward, done, info = self.task.get_reward(self.state, self.sim), self.is_terminal(), {}

        return self.state, reward, done, info

    def make_step(self, action=None):
        """Calculates new state.

        Args:
            action (np.array, optional): the agent's last action. Defaults to None.

        Returns:
            (np.array): agent's observation of the environment state
        """
        # take actions
        if action is not None:
            self.sim.set_property_values(self.task.action_var, action)

        # run simulation
        self.sim.run()

        return self.get_observation()

    def reset(self, init_conditions):
        """Resets the state of the environment and returns an initial observation.

        Args:
            init_conditions (np.array): the initial observation of the space.
        """
        self.current_step = 0
        self.close()

        self.sim = Simulation(
            aircraft_name=self.task.aircraft_name,
            init_conditions=init_conditions,
            jsbsim_freq=self.task.jsbsim_freq,
            agent_interaction_steps=self.task.agent_interaction_steps,
        )

        self.state = self.get_observation()

        return self.state

    def get_observation(self):
        """get state observation from sim.

        Returns:
            (NamedTuple): the first state observation of the episode
        """
        obs_list = self.sim.get_property_values(self.task.state_var)
        return tuple([np.array([obs]) for obs in obs_list])

    def get_sim_time(self):
        """ Gets the simulation time from sim, a float. """
        return self.sim.get_sim_time()

    def close(self):
        """Cleans up this environment's objects
        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sim:
            self.sim.close()

    def is_terminal(self):
        """Checks if the state is terminal.

        Returns:
            (bool)
        """
        is_not_contained = not self.observation_space.contains(self.state)

        return is_not_contained or self.task.is_terminal(self.state, self.sim)

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
        return self.task.render(self.sim, mode=mode, **kwargs)

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
