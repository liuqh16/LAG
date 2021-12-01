import numpy as np
from typing import List
from .env_base import BaseEnv
from ..core.catalog import Catalog
from ..core.simulatior import BaseSimulator
from ..tasks import HeadingTask

class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        assert self.num_aircrafts == 1, "only support one fighter"
        self._create_records = False

    @property
    def sims(self) -> List[BaseSimulator]:
        return list(self.jsbsims.values())

    def load_task(self):
        taskname = getattr(self.config, 'task', 'heading')
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def reset(self):
        self.current_step = 0
        self.heading_turns = 0
        self.reset_conditions()
        next_observation = self.get_observation()
        self.task.reset(self)
        return next_observation

    def reset_conditions(self):
        uid = list(self.aircraft_configs.keys())[0]
        new_init_state = self.aircraft_configs[uid].get('init_state', {})  # type: dict
        new_init_state.update({
            'ic_psi_true_deg': 0,
            'ic_u_fps': 800,
            'ic_v_fps': 0,
            'ic_w_fps': 0,
            'ic_p_rad_sec': 0,
            'ic_q_rad_sec': 0,
            'ic_r_rad_sec': 0,
            'target_heading_deg': 0,
            'target_altitude_ft': 20000,
            'target_velocities_u_mps': 243, 
            'heading_check_time': 20
        })
        self.sims[0].reload(new_init_state)

    def step(self, actions: list):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and 
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict{str: np.array}): the agents' action, with same length as action variables.

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward_visualize: amount of reward_visualize returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        
        actions = self.task.normalize_action(self, actions)
        for idx, sim in enumerate(self.jsbsims.values()):
            sim.set_property_values(self.task.action_var, actions[idx])
        # run simulator for one step
        for _ in range(self.agent_interaction_steps):
            for sim in self.sims:
                sim.run()

        next_observation = self.get_observation()
        rewards = np.zeros(self.num_aircrafts)
        for agent_id in range(self.num_aircrafts):
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        done = False
        for agent_id in range(self.num_aircrafts):
            agent_done, info = self.task.get_termination(self, agent_id, info)
            done = agent_done or done
        dones = done * np.ones(self.num_aircrafts)

        return next_observation, rewards, dones, info

    def get_observation(self):
        """
        get state observation from sim.

        Returns:
            (OrderedDict): the same format as self.observation_space
        """
        # generate observation (gym.Env output)
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

