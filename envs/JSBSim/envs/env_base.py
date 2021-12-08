import gym
from gym.utils import seeding
import numpy as np
from typing import Dict, List, Union
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
        self.max_steps = getattr(self.config, 'max_steps', 100)     # type: int
        self.jsbsim_freq = getattr(self.config, 'jsbsim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self.load()

    @property
    def agent_ids(self) -> List[str]:
        return self.__agent_ids

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self.__jsbsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.jsbsim_freq

    def load(self):
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def load_simulator(self):
        self.__jsbsims = {}   # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self.__jsbsims[uid] = AircraftSimulator(
                uid, config.get("color", "Red"),
                config.get("model", "f16"),
                config.get("init_state"),
                getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                self.jsbsim_freq)
        self.__agent_ids = list(self.__jsbsims.keys())

        for key, sim in self.__jsbsims.items():
            for k, s in self.__jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
        """
        # reset sim
        self.current_step = 0
        for sim in self.__jsbsims.values():
            sim.reload()
        # reset task
        self.task.reset(self)
        return self.get_obs()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information

        NOTE: shape of obs/rewards/dones: {agent_id: values}
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        for agent_id in self.agent_ids:
            action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self.__jsbsims.values():
                sim.run()

        obs = self.get_obs()

        rewards = {}
        for agent_id in self.agent_ids:
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        dones = {}
        for agent_id in range(self.num_agents):
            dones[agent_id], info = self.task.get_termination(self, agent_id, info)

        return obs, rewards, dones, info

    def get_obs_agent(self, agent_id: str):
        """Returns observation for agent_id.

        Returns:
            (np.array)
        """
        obs_agent = np.array(self.agents[agent_id].get_property_values(self.task.state_var))
        return self.task.normalize_obs(self, agent_id, obs_agent)

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        obs = dict([(agent_id, self.get_obs_agent(agent_id)) for agent_id in self.agent_ids])
        return obs

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        return np.stack([self.get_obs_agent(agent_id) for agent_id in self.agent_ids])

    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self.__jsbsims.values():
            sim.close()
        self.__jsbsims = {}

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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
