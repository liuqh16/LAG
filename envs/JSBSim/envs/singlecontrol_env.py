import numpy as np
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
        self.heading_turn_counts = 0
        self.reset_conditions()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_conditions(self):
        uid = list(self.aircraft_configs.keys())[0]
        new_init_state = self.aircraft_configs[uid].get('init_state', {})  # type: dict
        check_interval = self.aircraft_configs[uid].get('check_interval', 30)
        init_heading = np.random.uniform(0., 180.)
        init_altitude = np.random.uniform(14000., 30000.)
        init_velocities_u = np.random.uniform(400., 1200.)
        new_init_state.update({
            'ic_psi_true_deg': init_heading,
            'ic_h_sl_ft': init_altitude,
            'ic_u_fps': init_velocities_u,
            'target_heading_deg': init_heading,
            'target_altitude_ft': init_altitude,
            'target_velocities_u_mps': init_velocities_u * 0.3048,
            'heading_check_time': check_interval
        })
        self.agents[self.agent_ids[0]].reload(new_init_state)

    def _pack(self, data):
        """Pack single key-value dict into single value"""
        assert isinstance(data, dict)
        return np.array([data[self.agent_ids[0]]])

    def _unpack(self, data):
        """Unpack data into single key-value dict"""
        return {self.agent_ids[0]: data[0]}
