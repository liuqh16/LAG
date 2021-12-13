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

    def reset(self):
        self.current_step = 0
        self.heading_turns = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

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

    def _pack(self, data):
        """Pack single key-value dict into single value"""
        if isinstance(data, dict):
            return data[self.agent_ids[0]]
        else:
            return data

    def _unpack(self, data):
        """Unpack data into single key-value dict"""
        return {self.agent_ids[0]: data}
