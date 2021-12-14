import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU


class SingleCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self) -> int:
        return 2

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude (unit: °)
            c.position_lat_geod_deg,            # 1. latitude  (unit: °)
            c.position_h_sl_m,                  # 2. altitude  (unit: m)
            c.attitude_roll_rad,                # 3. roll      (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch     (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw       (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north   (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east    (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down    (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
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
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(18,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(18)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000             # 0. ego altitude  (unit: 5km)
        norm_obs[1] = np.linalg.norm(ego_feature[3:])    # 1. ego_v         (unit: mh)
        norm_obs[2] = ego_obs_list[8]                    # 2. ego_v_down    (unit: mh)
        norm_obs[3] = np.sin(ego_obs_list[3])            # 3. ego_roll_sin
        norm_obs[4] = np.cos(ego_obs_list[3])            # 4. ego_roll_cos
        norm_obs[5] = np.sin(ego_obs_list[4])            # 5. ego_pitch_sin
        norm_obs[6] = np.cos(ego_obs_list[4])            # 6. ego_pitch_cos
        norm_obs[7] = ego_obs_list[9] / 340              # 7. ego_vc        (unit: mh)
        norm_obs[8] = ego_obs_list[10]                   # 8. ego_north_ng  (unit: 5G)
        norm_obs[9] = ego_obs_list[11]                   # 9. ego_east_ng   (unit: 5G)
        norm_obs[10] = ego_obs_list[12]                  # 10. ego_down_ng   (unit: 5G)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[11] = R / 10000                         # 11. relative distance (unit: 10km)
        norm_obs[12] = ego_AO                            # 12. ego_AO        (unit: rad)
        norm_obs[13] = ego_TA                            # 13. ego_TA        (unit: rad)
        norm_obs[14] = side_flag                         # 14. enm_delta_heading: 1 or 0 or -1
        norm_obs[15] = enm_obs_list[2] / 5000            # 15. enm_altitude  (unit: 5km)
        norm_obs[16] = np.linalg.norm(enm_feature[3:])   # 16. enm_v         (unit: mh)
        norm_obs[17] = enm_obs_list[8]                   # 17. enm_v_down    (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info=...):
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info
