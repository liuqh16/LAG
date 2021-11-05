import numpy as np
from typing import List
from .env_base import BaseEnv
from ..core.catalog import Catalog
from ..core.simulatior import BaseSimulator
from ..tasks import SingleCombatTask, SingleCombatWithMissileTask, SingleCombatWithArtilleryTask, SingleCombatWithAvoidMissileTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self._create_records = False

    @property
    def sims(self) -> List[BaseSimulator]:
        return list(self.jsbsims.values()) + list(self.other_sims.values())

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'singlecombat_with_missile':
            self.task = SingleCombatWithMissileTask(self.config)
        elif taskname == 'singlecombat_with_avoid_missile':
            self.task = SingleCombatWithAvoidMissileTask(self.config)
        elif taskname == 'singlecombat_with_artillery':
            self.task = SingleCombatWithArtilleryTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        next_observation = self.get_observation()
        self.task.reset(self)
        return self._mask(next_observation)

    def reset_simulators(self):
        # Assign new initial condition here!
        for sim in self.jsbsims.values():
            sim.reload()
        self.other_sims = {}  # type: dict[str, BaseSimulator]

    def step(self, actions: list):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and 
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.array): the agents' action, with same length as action space.

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward_visualize: amount of reward_visualize returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}

        # apply action
        actions = self.task.normalize_action(self, actions)
        for idx, sim in enumerate(self.jsbsims.values()):
            sim.set_property_values(self.task.action_var, actions[idx])
        # run simulator for one step
        for _ in range(self.agent_interaction_steps):
            for sim in self.sims:
                sim.run()
        # call task.step for extra process
        self.task.step(self, actions)

        next_observation = self.get_observation()

        rewards = np.zeros(self.num_aircrafts)
        for agent_id in range(self.num_aircrafts):
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        done = False
        for agent_id in range(self.num_aircrafts):
            agent_done, info = self.task.get_termination(self, agent_id, info)
            done = agent_done or done
        dones = done * np.ones(self.num_aircrafts)

        return self._mask(next_observation), self._mask(rewards), self._mask(dones), info

    def get_observation(self):
        """
        get state observation from sim.

        Returns:
            (OrderedDict): the same format as self.observation_space
        """
        next_observation = []
        for sim in self.jsbsims.values():
            next_observation.append(sim.get_property_values(self.task.state_var))
        next_observation = self.task.normalize_observation(self, next_observation)
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
                timestamp = self.current_step * self.agent_interaction_steps / self.jsbsim_freq
                f.write(f"#{timestamp:.2f}\n")
                for sim in self.sims:
                    f.write(sim.log() + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def seed(self, seed):
        # TODO: random seed
        return super().seed(seed=seed)

    def _mask(self, data):
        return np.expand_dims(data[0], axis=0) if self.task.use_baseline else data
