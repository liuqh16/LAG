import numpy as np
from typing import List
from .env_base import BaseEnv
from ..core.simulatior import AircraftSimulator
from ..tasks.heading_task import HeadingTask, HeadingAndAltitudeTask, HeadingContinuousTask


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.jsbsims) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"

    @property
    def agent(self) -> AircraftSimulator:
        return self.jsbsims[0]

    @property
    def sims(self) -> List[AircraftSimulator]:
        return self.jsbsims

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading_task':
            self.task = HeadingTask(self.config)
        elif taskname == 'heading_altitude_task':
            self.task = HeadingAndAltitudeTask(self.config)
        elif taskname == 'heading_continuous_task':
            self.task = HeadingContinuousTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def load_simulator(self):
        self.jsbsims = []   # type: List[AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self.jsbsims.append(
                AircraftSimulator(uid, config.get("team", "f16"), config.get("model", "f16"), config.get("init_state"),
                                  getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)), self.jsbsim_freq)
            )

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        return self.get_obs()

    def reset_simulators(self):
        new_init_state = self.jsbsims[0].init_state
        new_init_state.update({
            'ic_psi_true_deg': self.np_random.uniform(0, 360),
            'ic_u_fps': self.np_random.uniform(500, 1000),
            'ic_v_fps': self.np_random.uniform(-100, 100),
            'ic_w_fps': self.np_random.uniform(-100, 100),
            'ic_p_rad_sec': self.np_random.uniform(-np.pi, np.pi),
            'ic_q_rad_sec': self.np_random.uniform(-np.pi, np.pi),
            'ic_r_rad_sec': self.np_random.uniform(-np.pi, np.pi),
        })
        self.jsbsims[0].reload(new_init_state)

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

        # apply actions
        action = self.task.normalize_action(self, action)
        self.agent.set_property_values(self.task.action_var, action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            self.agent.run()

        obs = self.get_obs()

        reward = self.task.get_reward(self, 0, info)

        done, info = self.task.get_termination(self, 0, info)

        return obs, reward, done, info

    def get_obs(self):
        obs = self.agent.get_property_values(self.task.state_var)
        return self.task.normalize_obs(self, obs)
