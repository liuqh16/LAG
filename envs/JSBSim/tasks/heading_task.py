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
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  #  3. altitude  (unit: m)
            c.attitude_roll_rad,                #  4. roll      (unit: rad)
            c.attitude_pitch_rad,               #  5. pitch     (unit: rad)
            c.velocities_u_mps,                 #  6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  9. vc        (unit: m/s)
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
        self.observation_space = [spaces.Box(low=-10, high=10., shape=(12,)) for _ in range(self.num_agents)]

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]

    def normalize_observation(self, env, obsersvations: list):
        """
        Convert simulation states into the format of observation_space.

            observation(dim 12):

            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad) 
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        ego_obs_list = obsersvations[0]
        observation = np.zeros(12)
        observation[0] = ego_obs_list[0] / 1000     
        observation[1] = ego_obs_list[1] / 180 * np.pi  
        observation[2] = ego_obs_list[2] / 340         
        observation[3] = ego_obs_list[3] / 5000       
        observation[4] = np.sin(ego_obs_list[4])     
        observation[5] = np.cos(ego_obs_list[4])
        observation[6] = np.sin(ego_obs_list[5])
        observation[7] = np.cos(ego_obs_list[5])
        observation[8] = ego_obs_list[6] / 340
        observation[9] = ego_obs_list[7] / 340
        observation[10] = ego_obs_list[8] / 340
        observation[11] = ego_obs_list[9] / 340
        observation = np.expand_dims(observation, axis=0)  # dim: (1,12)
        return observation

    def normalize_action(self, env, action: list):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros((1, 4))
        norm_act[0, 0] = action[0][0] * 2. / (self.action_space[0].nvec[0] - 1.) - 1.
        norm_act[0, 1] = action[0][1] * 2. / (self.action_space[0].nvec[1] - 1.) - 1.
        norm_act[0, 2] = action[0][2] * 2. / (self.action_space[0].nvec[2] - 1.) - 1.
        norm_act[0, 3] = action[0][3] * 0.5 / (self.action_space[0].nvec[3] - 1.) + 0.4
        return norm_act
