import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading, UnreachHeadingAndAltitude


class HeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):
        self.config = config
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
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_θ   (unit: °)
            c.attitude_roll_rad,                # 2. roll      (unit: rad)
            c.attitude_pitch_rad,               # 3. pitch     (unit: rad)
            c.velocities_v_north_mps,           # 4. v_north   (unit: mps)
            c.velocities_v_east_mps,            # 5. v_east    (unit: mps)
            c.velocities_v_down_mps,            # 6. v_down    (unit: mps)
            c.velocities_vc_mps,                # 7. vc        (unit: mps)
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
        self.observation_space = spaces.Box(low=-10, high=10., shape=(8,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def normalize_obs(self, env, obs):
        """Normalize raw observation to make training easier.
        """
        norm_obs = np.zeros(8)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude  (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading   (unit rad)
        norm_obs[2] = obs[2]                # 2. ego_roll      (unit: rad)
        norm_obs[3] = obs[3]                # 3. ego_pitch     (unit: rad)
        norm_obs[4] = obs[4] / 340          # 4. ego_v_north   (unit: mh)
        norm_obs[5] = obs[5] / 340          # 5. ego_v_east    (unit: mh)
        norm_obs[6] = obs[6] / 340          # 6. ego_v_down    (unit: mh)
        norm_obs[7] = obs[7] / 340          # 7. ego_vc        (unit: mh)
        return norm_obs

    def normalize_action(self, env, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space[0].nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space[0].nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space[0].nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space[0].nvec[3] - 1.) + 0.4
        return norm_act


class HeadingContinuousTask(HeadingTask):
    '''
    Control target heading with continuous action space
    '''
    def __init__(self, config):
        super().__init__(config)

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, 0.4], dtype=np.float32),
                                       high=np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32))

    def normalize_action(self, env, action):
        """Clip continuous value into proper value.
        """
        return np.clip(action, self.action_space.low, self.action_space.high)


class HeadingAndAltitudeTask(HeadingTask):
    '''
    Control target heading and target altitude with discrete action space
    '''
    def __init__(self, config):
        super().__init__(config)
        self.termination_conditions = [
            UnreachHeadingAndAltitude(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
