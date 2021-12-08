import gym
from gym.utils import seeding
import numpy as np
from typing import Dict, List
from ..core.simulatior import AircraftSimulator, BaseSimulator
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
        self._create_records = False
        self.load()

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.jsbsim_freq

    @property
    def observation_space(self) -> Dict[str, gym.Space]:
        return self.task.observation_space

    @property
    def action_space(self) -> Dict[str, gym.Space]:
        return self.task.action_space

    def load(self):
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)

    def load_simulator(self):
        self._jsbsims = {}     # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self._jsbsims[uid] = AircraftSimulator(
                uid, config.get("color", "Red"),
                config.get("model", "f16"),
                config.get("init_state"),
                getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                self.jsbsim_freq)
        self._agent_ids = list(self._jsbsims.keys())

        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

        self._tempsims = {}    # type: Dict[str, BaseSimulator]

    def add_temp_simulator(self, sim: BaseSimulator):
        self._tempsims[sim.uid] = sim

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
        """
        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
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
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():
                sim.run()
            for sim in self._tempsims.values():
                sim.run()
        self.task.step(self)

        obs = self.get_obs()

        rewards = {}
        for agent_id in self.agent_ids:
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        dones = {}
        for agent_id in self.agent_ids:
            dones[agent_id], info = self.task.get_termination(self, agent_id, info)

        return obs, rewards, dones, info

    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id in self.agent_ids])

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        return np.stack([self.task.get_obs(self, agent_id) for agent_id in self.agent_ids])

    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
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
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for sim in self._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

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
