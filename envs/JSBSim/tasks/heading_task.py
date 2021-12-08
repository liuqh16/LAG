import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading


class HeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h    (unit: m)
            c.delta_heading,                    # 1. delta_θ    (unit: °)
            c.delta_velocities_u,               # 2. delta_v    (unit: m/s)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.velocities_u_mps,                 # 5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 8. vc         (unit: m/s)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(11,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def normalize_obs(self, env, agent_id, obs):
        """Normalize raw observation to make training easier.
        """
        norm_obs = np.zeros(11)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude  (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading   (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = np.sin(obs[3])        # 3. ego_roll_sin
        norm_obs[4] = np.cos(obs[3])        # 4. ego_roll_cos
        norm_obs[5] = np.sin(obs[4])        # 5. ego_pitch_sin
        norm_obs[6] = np.cos(obs[4])        # 6. ego_pitch_cos
        norm_obs[7] = obs[5] / 340          # 4. ego_v_north   (unit: mh)
        norm_obs[8] = obs[6] / 340          # 5. ego_v_east    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 6. ego_v_down    (unit: mh)
        norm_obs[10] = obs[8] / 340         # 7. ego_vc        (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[0] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[0] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[0] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act
