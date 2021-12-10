import numpy as np
from gym import spaces
from typing import Tuple
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU


class MultipleCombatTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            AltitudeReward(self.config),
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
        return 4

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
        self.obs_length = 11 + (self.num_agents - 1) * 7
        self.observation_space = dict([(agend_id, spaces.Box(low=-10, high=10., shape=(self.obs_length,))) for agend_id in self.agent_ids])
        self.share_observation_space = dict([(agend_id, spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))) for agend_id in self.agent_ids])

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = dict([(agend_id, spaces.MultiDiscrete([41, 41, 41, 30])) for agend_id in self.agent_ids])

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(self.obs_length)
        # (1) ego info normalization
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        ego_cur_ned = LLA2NEU(*ego_state[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_state[6:9])])
        norm_obs[0] = ego_state[2] / 5000               # 0. ego altitude  (unit: 5km)
        norm_obs[1] = np.linalg.norm(ego_feature[3:])   # 1. ego_v         (unit: mh)
        norm_obs[2] = ego_state[8]                      # 2. ego_v_down    (unit: mh)
        norm_obs[3] = np.sin(ego_state[3])              # 3. ego_roll_sin
        norm_obs[4] = np.cos(ego_state[3])              # 4. ego_roll_cos
        norm_obs[5] = np.sin(ego_state[4])              # 5. ego_pitch_sin
        norm_obs[6] = np.cos(ego_state[4])              # 6. ego_pitch_cos
        norm_obs[7] = ego_state[9] / 340                # 7. ego_vc        (unit: mh)
        norm_obs[8] = ego_state[10]                     # 8. ego_north_ng  (unit: 5G)
        norm_obs[9] = ego_state[11]                     # 9. ego_east_ng   (unit: 5G)
        norm_obs[10] = ego_state[12]                    # 10. ego_down_ng   (unit: 5G)
        # (2) relative inof w.r.t partner+enemies state
        offset = 10
        for sim in env.agents[agent_id].partners + env.agents[agent_id].enemies:
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], env.center_lon, env.center_lat, env.center_alt)
            feature = np.array([*cur_ned, *(state[6:9])])
            AO, TA, R, side_flag = get_AO_TA_R(ego_feature, feature, return_side=True)
            norm_obs[offset + 1] = R / 10000                    # relative distance (unit: 10km)
            norm_obs[offset + 2] = AO                           # AO        (unit: rad)
            norm_obs[offset + 3] = TA                           # TA        (unit: rad)
            norm_obs[offset + 4] = side_flag                    # delta_heading: 1 or 0 or -1
            norm_obs[offset + 5] = state[2] / 5000              # altitude  (unit: 5km)
            norm_obs[offset + 6] = np.linalg.norm(feature[3:])  # v         (unit: mh)
            norm_obs[offset + 7] = state[8]                     # v_down    (unit: mh)
            offset += 7
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space[agent_id].nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space[agent_id].nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space[agent_id].nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space[agent_id].nvec[3] - 1.) + 0.4
        return norm_act

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info
