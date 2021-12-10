import numpy as np
from typing import Tuple, Dict, Sequence
from .env_base import BaseEnv
from ..tasks.multiplecombat_task import MultipleCombatTask


class MultipleCombatEnv(BaseEnv):
    """
    MultipleCombatEnv is an multi-player competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False

    @property
    def share_observation_space(self):
        return self.task.share_observation_space

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'multiplecombat':
            self.task = MultipleCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (dict): {agent_id: initial observation}
            share_obs (dict): {agent_id: initial state}
        """
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        return self.get_obs(), self.get_state()

    def reset_simulators(self):
        # Assign new initial condition here!
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()

    def step(self, action: Dict[str, Sequence]) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict): the agents' actions, each key corresponds to an agent_id

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                share_obs: agents' share observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
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
        share_obs = self.get_state()

        rewards = {}    # type: Dict[str, float]
        for agent_id in self.agent_ids:
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        dones = {}      # type: Dict[str, bool]
        for agent_id in self.agent_ids:
            dones[agent_id], info = self.task.get_termination(self, agent_id, info)

        return obs, share_obs, rewards, dones, info
