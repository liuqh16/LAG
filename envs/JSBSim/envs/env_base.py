import gym
import numpy as np
from ..core.simulatior import AircraftSimulator
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
    metadata = {"render.modes": ["human", "txt"]}

    def __init__(self, config_name: str):
        self.config = parse_config(config_name)
        self.aircraft_configs = self.config.aircraft_configs     # type: dict
        self.num_aircrafts = len(self.aircraft_configs.keys())
        self.max_steps = getattr(self.config, 'max_steps', 100)   # type: int
        self.jsbsim_freq = getattr(self.config, 'jsbsim_freq', 60)   # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.load()

    @property
    def num_agents(self):
        return self.task.num_agents

    @property
    def time_interval(self):
        return self.agent_interaction_steps / self.jsbsim_freq

    def load(self):
        self.load_task()
        self.load_variables()
        self.load_simulator()

    def load_task(self):
        self.task = BaseTask(self.config)
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def load_variables(self):
        self.current_step = 0
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))

    def load_simulator(self):
        self.jsbsims = dict([(uid, AircraftSimulator(
            uid=uid,
            team=self.aircraft_configs[uid].get("team", "Red"),
            model=self.aircraft_configs[uid].get("model", "f16"),
            init_state=self.aircraft_configs[uid].get("init_state", {}),
            origin=(self.center_lon, self.center_lat, self.center_alt),
            jsbsim_freq=self.jsbsim_freq)) for uid in self.aircraft_configs.keys()])

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Args:
            init_conditions (np.array): the initial observation of the space.
        """
        self.current_step = 0
        for sim in self.jsbsims.values():
            sim.reload()
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

        # take actions
        for _ in range(self.agent_interaction_steps):
            for idx, sim in enumerate(self.jsbsims.values()):
                sim.set_property_values(self.task.action_var, actions[idx])
                sim.run()

        next_observation = self.get_observation()
        reward, info = self.task.get_reward(self, 0, info)
        done, info = self.task.get_termination(self, 0, info)

        return next_observation, reward, done, info

    def get_observation(self):
        """get state observation from sim.

        Returns:
            (np.array): the first state observation of the episode
        """
        next_observation = []
        for sim in self.jsbsims.values():
            next_observation.append(sim.get_property_values(self.task.state_var))
        return np.array(next_observation)

    def close(self):
        """Cleans up this environment's objects
        Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self.jsbsims.values():
            if sim:
                sim.close()

    def render(self, mode="human", **kwargs):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files

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
