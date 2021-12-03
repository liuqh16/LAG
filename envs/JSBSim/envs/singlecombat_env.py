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
        assert len(self.jsbsims.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self._create_records = False

    @property
    def sims(self) -> Dict[str, BaseSimulator]:
        sims = {}
        sims.update(self.jsbsims)
        sims.update(self.other_sims)
        return sims

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
        next_observation = self.get_observation()
        return self._mask(next_observation)

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
        for sim in [sim for uid, sim in self.jsbsims.items() if uid[0] != self.ego_team]:
            action = self.task.rollout(self, sim)
            sim.set_property_values(self.task.action_space, action)
        # run simulator for one step
        for _ in range(self.agent_interaction_steps):
            for sim in self.sims.values():
                sim.run()
        # call task.step for extra process
        self.task.step(self, action)

        next_observation = self.get_observation()

        reward = np.zeros(self.num_agents)
        for agent_id in range(self.num_agents):
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        done = False
        for agent_id in range(self.num_agents):
            agent_done, info = self.task.get_termination(self, agent_id, info)
            done = agent_done or done
        dones = done * np.ones(self.num_agents)

        return self._mask(next_observation), self._mask(rewards), self._mask(dones), info

    def get_obs_agent(self, agent_id: int):
        return super().get_obs_agent(agent_id)

    def get_observation(self):
        """
        get state observation from sim.

        Returns:
            (OrderedDict): the same format as self.observation_space
        """
        next_observation = []
        for sim in self.jsbsims.values():
            next_observation.append(sim.get_property_values(self.task.state_var))
        next_observation = self.task.normalize_obs(self, next_observation)
        return next_observation

    def close(self):
        """Cleans up this environment's objects.

        Environments automatically close() when garbage collected or when the program exits.
        """
        for sim in self.sims:
            sim.close()

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
                for sim in self.sims:
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def _mask(self, data):
        return np.expand_dims(data[0], axis=0) if self.task.use_baseline else data
