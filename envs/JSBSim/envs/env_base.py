import gym
from gym.utils import seeding
import numpy as np
from typing import Dict, List
from ..core.simulatior import BaseSimulator, AircraftSimulator
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
        self._ego_team = None
        self._ego_sims = []
        self.load()

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def agents(self) -> List[AircraftSimulator]:
        return self._ego_sims

    @property
    def sims(self) -> Dict[str, BaseSimulator]:
        sims = {}
        sims.update(self.jsbsims)
        return sims

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
        self.jsbsims = {}   # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            if self._ego_team is None:
                self._ego_team = uid[0]
            self.jsbsims[uid] = AircraftSimulator(
                uid, config.get("color", "Red"),
                config.get("model", "f16"),
                config.get("init_state"),
                getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                self.jsbsim_freq)
            if uid[0] == self._ego_team:
                self._ego_sims.append(self.jsbsims[uid])

        for key, sim in self.jsbsims.items():
            for k, s in self.jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Args:
            init_conditions (np.array): the initial observation of the space.
        """
        # reset sim
        self.current_step = 0
        for sim in self.jsbsims.values():
            sim.reload()
        # reset task
        self.task.reset(self)
        # return obs[0]
        init_obs = self.get_obs()
        return init_obs

    def step(self, actions):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            actions (np.array): the agents' actions, with same length as num_agents

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information

        NOTE: shape of obs/rewards/dones: [num_agents, *dim]
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply actions
        for agent_id in range(self.num_agents):
            action = self.task.normalize_action(self, actions[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, action)
        for sim in [sim for uid, sim in self.jsbsims.items() if uid[0] != self._ego_team]:
            action = self.task.rollout(self, sim)
            sim.set_property_values(self.task.action_space, action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self.sims.values():
                sim.run()

        obs = self.get_obs()

        rewards = np.zeros((self.num_agents, 1))
        for agent_id in range(self.num_agents):
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        dones = np.zeros((self.num_agents, 1))
        for agent_id in range(self.num_agents):
            dones[agent_id], info = self.task.get_termination(self, agent_id, info)

        return obs, rewards, dones, info

    def get_obs_agent(self, agent_id: int):
        """Returns observation for agent_id.

        Returns:
            (np.array)
        """
        return np.array(self.agents[agent_id].get_property_values(self.task.state_var))

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.num_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        pass

    def load_policy(self, name: str):
        """Load a specific strategy for opponents

        Args:
            name (str): Baseline name or Model path
        """
        self.task.load_policy(self, name)

    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self.sims.values():
            sim.close()
        self.jsbsims = {}

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
