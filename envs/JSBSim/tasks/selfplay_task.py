import numpy as np
from collections import OrderedDict
from gym import spaces
from torch import ones, pdist
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, PostureReward, RelativeAltitudeReward, SmoothActionReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..utils.utils import lonlat2dis, get_AO_TA_R


class SelfPlayTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
            SmoothActionReward(self.config),
        ]

        self.termination_conditions = [
            ShootDown(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

        self.bloods = dict([(agent, 100) for agent in self.config.init_config.keys()])
        self.init_longitude, self.init_latitude = 0, 0

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             #  0. lontitude (unit: degree)
            c.position_lat_geod_deg,            #  1. latitude  (unit: degree)
            c.position_h_sl_ft,                 #  2. altitude  (unit: feet)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.attitude_heading_true_rad,        #  5. yaw       (unit: rad)
            c.velocities_v_north_fps,           #  6. v_north   (unit: fps)
            c.velocities_v_east_fps,            #  7. v_east    (unit: fps)
            c.velocities_v_down_fps,            #  8. v_down    (unit: fps)
            c.velocities_vc_fps,                #  9. vc        (unit: fps)
            c.accelerations_n_pilot_x_norm,     # 10. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 11. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 12. a_down    (unit: G)
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
            c.position_h_sl_ft,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        space_dict = OrderedDict()
        for fighter_name in list(self.config.init_config.keys()):
            space_dict[fighter_name] = spaces.Dict(
                        OrderedDict({
                            'ego_info': spaces.Box(low=-10, high=10., shape=(18,)),
                        }))
        self.observation_space = spaces.Dict(space_dict)

    def load_action_space(self):
        space_dict = OrderedDict({
                "aileron": spaces.Discrete(41),
                "elevator": spaces.Discrete(41),
                'rudder': spaces.Discrete(41),
                "throttle": spaces.Discrete(30),
            })
        self.action_space = spaces.Dict(space_dict)

    def normalize_observation(self, env, sorted_all_obs_list: list):
        ego_obs_list, enm_obs_list = sorted_all_obs_list[0], sorted_all_obs_list[1]
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_east, ego_cur_north = lonlat2dis(ego_obs_list[0], ego_obs_list[1], env.init_longitude, env.init_latitude)
        enm_cur_east, enm_cur_north = lonlat2dis(enm_obs_list[0], enm_obs_list[1], env.init_longitude, env.init_latitude)
        ego_feature = np.array([
            ego_cur_north / 1000, ego_cur_east / 1000, ego_obs_list[2] * 0.304 / 1000,
            ego_obs_list[6] * 0.304 / 340, ego_obs_list[7] * 0.304 / 340, ego_obs_list[8] * 0.304 / 340
        ])
        enm_feature = np.array([
            enm_cur_north / 1000, enm_cur_east / 1000, enm_obs_list[2] * 0.304 / 1000,
            enm_obs_list[6] * 0.304 / 340, enm_obs_list[7] * 0.304 / 340, enm_obs_list[8] * 0.304 / 340
        ])
        observation = np.zeros(18)
        # (1) ego info normalization
        observation[0] = ego_obs_list[2] * 0.304 / 5000     #  0. ego altitude  (unit: 5km)
        observation[1] = np.linalg.norm(ego_feature[3:])    #  1. ego_v         (unit: mh)
        observation[2] = ego_obs_list[8]                    #  2. ego_v_down    (unit: mh)
        observation[3] = np.sin(ego_obs_list[3])            #  3. ego_roll_sin
        observation[4] = np.cos(ego_obs_list[3])            #  4. ego_roll_cos
        observation[5] = np.sin(ego_obs_list[4])            #  5. ego_pitch_sin
        observation[6] = np.cos(ego_obs_list[4])            #  6. ego_pitch_cos
        observation[7] = ego_obs_list[9] * 0.304 / 340      #  7. ego_vc        (unit: mh)
        observation[8] = ego_obs_list[10]                   #  8. ego_north_ng  (unit: 5G)
        observation[9] = ego_obs_list[11]                   #  9. ego_east_ng   (unit: 5G)
        observation[10] = ego_obs_list[12]                  # 10. ego_down_ng   (unit: 5G)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        observation[11] = R / 10                            # 11. relative distance (unit: 10km)
        observation[12] = ego_AO                            # 12. ego_AO        (unit: rad)
        observation[13] = ego_TA                            # 13. ego_TA        (unit: rad)
        observation[14] = side_flag                         # 14. enm_delta_heading: 1 or 0 or -1
        observation[15] = enm_obs_list[2] * 0.304 / 5000    # 15. enm_altitude  (unit: 5km)
        observation[16] = np.linalg.norm(enm_feature[3:])   # 16. enm_v         (unit: mh)
        observation[17] = enm_obs_list[8]                   # 17. enm_v_down    (unit: mh)
        return observation

    # def normalize_observation(self, env, sorted_all_obs_list: list):
    #     ego_obs_list, enm_obs_list = sorted_all_obs_list[0], sorted_all_obs_list[1]
    #     observation = np.zeros(22)
    #     ego_cur_east, ego_cur_north = lonlat2dis(ego_obs_list[0], ego_obs_list[1], env.init_longitude, env.init_latitude)
    #     enm_cur_east, enm_cur_north = lonlat2dis(enm_obs_list[0], enm_obs_list[1], env.init_longitude, env.init_latitude)
    #     observation[0] = ego_cur_north / 10000.             # (1) ego north, unit: 10km
    #     observation[1] = ego_cur_east / 10000.              # (2) ego east, unit: 10km
    #     observation[2] = ego_obs_list[2] * 0.304 / 5000     # (3) ego altitude, unit: 5km
    #     observation[3] = np.cos(ego_obs_list[3])            # (4~9) ego cos/sin of roll, pitch, yaw
    #     observation[4] = np.sin(ego_obs_list[3])
    #     observation[5] = np.cos(ego_obs_list[4])
    #     observation[6] = np.sin(ego_obs_list[4])
    #     observation[7] = np.cos(ego_obs_list[5])
    #     observation[8] = np.sin(ego_obs_list[5])
    #     observation[9] = ego_obs_list[6] * 0.304 / 340      # (10) ego v_n, unit: mh
    #     observation[10] = ego_obs_list[7] * 0.304 / 340     # (11) ego v_e, unit: mh
    #     observation[11] = ego_obs_list[8] * 0.304 / 340     # (12) ego v_d, unit: mh
    #     observation[12] = ego_obs_list[9] * 0.304 / 340     # (13) ego vc, unit: mh
    #     observation[13] = ego_obs_list[10] / 5              # (14~16) ego accelaration, unit: G
    #     observation[14] = ego_obs_list[11] / 5
    #     observation[15] = ego_obs_list[12] / 5
    #     observation[16] = enm_cur_north / 10000.            # (17) enm north, unit: 10km
    #     observation[17] = enm_cur_east / 10000.             # (18) enm east, unit: 10km
    #     observation[18] = enm_obs_list[2] * 0.304 / 5000    # (19) enm altitude, unit: 5km
    #     observation[19] = enm_obs_list[6] * 0.304 / 340     # (20~22) enm v(NED), unit: mh
    #     observation[20] = enm_obs_list[7] * 0.304 / 340
    #     observation[21] = enm_obs_list[8] * 0.304 / 340
    #     return observation

    def process_actions(self, env, action: dict):
        """Convert discrete action index into continuous value.
        """
        for agent_name in env.agent_names:
            action[agent_name] = np.array(action[agent_name], dtype=np.float32)
            action[agent_name][0] = action[agent_name][0] * 2. / (self.action_space['aileron'].n - 1.) - 1.
            action[agent_name][1] = action[agent_name][1] * 2. / (self.action_space['elevator'].n - 1.) - 1.
            action[agent_name][2] = action[agent_name][2] * 2. / (self.action_space['rudder'].n - 1.) - 1.
            action[agent_name][3] = action[agent_name][3] * 0.5 / (self.action_space['throttle'].n - 1.) + 0.4
        return action

    def reset(self, env):
        """Task-specific reset, include reward function reset.

        Must call it after `env.get_observation()`
        """
        self.bloods = dict([(agent, 100) for agent in env.agent_names])
        return super().reset(env)

    def get_reward(self, env, agent_id, info={}):
        """
        Must call it after `env.get_observation()`
        """
        return super().get_reward(env, agent_id, info)

    def get_termination(self, env, agent_id, info={}):
        return super().get_termination(env, agent_id, info)
