import numpy as np
from typing import List, Dict
from .env_base import BaseEnv
from ..core.simulatior import BaseSimulator
from ..tasks.singlecombat_task import SingleCombatTask
from ..tasks.singlecombat_with_missle_task import SingleCombatWithMissileTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agent_ids) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self._create_records = False
        self.__tempsims = {}  # type: Dict[str, BaseSimulator]

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'singlecombat_with_missile':
            self.task = SingleCombatWithMissileTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        return self.get_obs()

    def reset_simulators(self):
        # switch side
        init_states = [sim.init_state for sim in self.jsbsims.values()]
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.jsbsims.values()):
            sim.reload(init_states[idx])
        self.other_sims = {}  # type: Dict[str, BaseSimulator]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.array): the agent's action, with same length as action variables.

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward: amount of reward returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply action
        action = self.task.normalize_action(self, action)
        self.agents[0].set_property_values(self.task.action_var, action)
        for sim in [sim for uid, sim in self.jsbsims.items() if uid[0] != self._ego_team]:
            action = self.task.rollout(self, sim)
            sim.set_property_values(self.task.action_var, action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self.sims.values():
                sim.run()
        # call task.step for extra process
        self.task.step(self, action)

        obs = self.get_obs()

        reward = self.task.get_reward(self, 0, info)

        done, info = self.task.get_termination(self, 0, info)

        return obs, reward, done, info

    def get_obs_agent(self, agent_id: str):
        obs_ego = np.array(self.agents[agent_id].get_property_values(self.task.state_var))
        # select the first enemy's state as extra input
        obs_enemy = np.array(self.agents[agent_id].enemies[0].get_property_values(self.task.state_var))
        return self.task.normalize_obs(self, agent_id, np.hstack((obs_ego, obs_enemy)))

    def close(self):
        res = super().close()
        self.other_sims = {}
        return res

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
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
                for sim in self.sims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError
