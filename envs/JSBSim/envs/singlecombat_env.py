import numpy as np
from .env_base import BaseEnv
from ..core.catalog import Catalog
from ..core.simulatior import AircraftSimulator
from ..tasks import SingleCombatTask, SingleCombatWithMissileTask, SingleCombatWithArtilleryTask, SingleCombatWithAvoidMissileTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!

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
        self.reset_conditions()
        next_observation = self.get_observation()
        self.task.reset(self)
        return self._mask(next_observation)

    def reset_conditions(self):
        # Assign new initial condition here!
        for idx in range(self.num_aircrafts):
            self.sims[idx].reload()

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
        info = {}
        actions = self.task.normalize_action(self, actions)
        # run JSBSim for one step
        next_observation = self.make_step(actions)
        # call task.step for extra simulation
        self.task.step(self, actions)

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
        for agent_id in range(self.num_aircrafts):
            next_observation.append(self.sims[agent_id].get_property_values(self.task.state_var))
        next_observation = self.task.normalize_observation(self, next_observation)
        return next_observation

    def close(self):
        """Cleans up this environment's objects.

        Environments automatically close() when garbage collected or when the program exits.
        """
        for agent_id in range(self.num_aircrafts):
            if self.sims[agent_id]:
                self.sims[agent_id].close()

    def render(self, mode="human"):
        # TODO: real time rendering [Use FlightGear]
        render_list = []
        for agent_id in range(self.num_aircrafts):
            # flight
            render_list.append(np.array(self.sims[agent_id].get_property_values(self.task.render_var)))
            # missile
            if getattr(self.task, 'missile_lists', None) is not None:
                render_list.extend(self.task.get_missile_trajectory(self, agent_id))
        return np.hstack(render_list)

    def seed(self, seed):
        # TODO: random seed
        return super().seed(seed=seed)

    def _mask(self, data):
        return np.expand_dims(data[0], axis=0) if self.task.use_baseline else data
