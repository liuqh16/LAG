import gym
import numpy as np
from collections import OrderedDict
from ..core.simulation import Simulation
from ..tasks.task_base import BaseTask


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

    def __init__(self, task, config):
        assert isinstance(task, BaseTask), "TypeError: must give a instance of BaseTask"
        self.task = task
        self.config = self.task.config

        # env config
        self.max_steps = getattr(self.config, 'max_steps', 100)
        self.observation_space = self.task.get_observation_space()
        self.action_space = self.task.get_action_space()

        # agent config
        assert isinstance(getattr(self.config, 'init_config', None), dict) \
            and isinstance(list(self.config.init_config.values())[0], dict), \
            "Unexpected config error!"
        self.agent_names = list(self.config.init_config.keys())
        self.num_agents = len(self.agent_names)

        # simulation config
        self.jsbsim_freq = self.config.jsbsim_freq
        self.agent_interaction_steps = self.config.agent_interaction_steps
        self.aircraft_names = OrderedDict(  # aircraft model (Default: f16)
           [(agent, self.config.init_config[agent].get('aircraft_name', 'f16')) for agent in self.agent_names]
        )

        # custom config
        self.init_variables()

    def init_variables(self):
        self.current_step = 0
        self.sims = OrderedDict([(agent, None) for agent in self.agent_names])
        self.init_longitude, self.init_latitude = 0.0, 0.0
        self.init_conditions = OrderedDict([(agent, None) for agent in self.agent_names])
        self.state = None

    def reset(self, init_conditions):
        """Resets the state of the environment and returns an initial observation.

        Args:
            init_conditions (np.array): the initial observation of the space.
        """
        self.current_step = 0
        self.close()

        self.sims[0] = Simulation(
            aircraft_name=self.aircraft_names[self.agent_names[0]],
            init_conditions=self.init_conditions[self.agent_names[0]],
            jsbsim_freq=self.jsbsim_freq,
            agent_interaction_steps=self.agent_interaction_steps,
            origin_lon=self.init_longitude,
            origin_lat=self.init_latitude
        )

        self.state = self.get_observation()

        return self.state

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
        info = {}
        if action is not None:
            if not len(action) == len(self.action_space.spaces):
                raise ValueError("mismatch between action and action space size")

        self.state = self.make_step(action)

        reward, info = self.task.get_reward(self, 0, info)
        done, info = self.task.get_termination(self, 0, info)

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
            self.sims[0].set_property_values(self.task.action_var, action)

        # run simulation
        self.sims[0].run()

        return self.get_observation()

    def get_observation(self):
        """get state observation from sim.

        Returns:
            (NamedTuple): the first state observation of the episode
        """
        obs_list = self.sims[0].get_property_values(self.task.state_var)
        return tuple([np.array([obs]) for obs in obs_list])

    def get_sim_time(self):
        """ Gets the simulation time from sim, a float. """
        return self.sims[0].get_sim_time()

    def close(self):
        """Cleans up this environment's objects
        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sims[0]:
            self.sims[0].close()

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
