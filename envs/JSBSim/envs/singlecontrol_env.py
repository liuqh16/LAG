from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agent_ids) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def reset(self):
        self.current_step = 0
        self.heading_turns = 0
        self.reset_simulators()
        self.task.reset(self)
        return self.get_obs()

    def reset_simulators(self):
        new_init_state = self.agents[self.agent_ids[0]].init_state
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
        self.agents[self.agent_ids[0]].reload(new_init_state)

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

        agent_id = self.agent_ids[0]
        # apply actions
        action = self.task.normalize_action(self, agent_id, action)
        self.agents[agent_id].set_property_values(self.task.action_var, action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            self.agents[agent_id].run()

        obs = self.get_obs()

        reward, info = self.task.get_reward(self, agent_id, info)

        done, info = self.task.get_termination(self, agent_id, info)

        return obs, reward, done, info

    def get_obs(self):
        return self.get_obs_agent(self.agent_ids[0])
