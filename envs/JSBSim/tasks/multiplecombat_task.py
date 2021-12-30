import torch
import numpy as np
from gym import spaces
from typing import Tuple
from .task_base import BaseTask
from ..core.simulatior import AircraftSimulator
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


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

        self.baseline_agents = {}   # will be initialized on first call of self.normalize_action

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
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

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
        # 1. Baseline action
        if agent_id not in self.baseline_agents:
            self.baseline_agents[agent_id] = Baseline2v2Agent()
        baseline_action = self.baseline_agents[agent_id].get_action(env.agents[agent_id])

        # # 2. RL action
        # norm_act = np.zeros(4)
        # norm_act[0] = action[0] * 2. / (self.action_space[agent_id].nvec[0] - 1.) - 1.
        # norm_act[1] = action[1] * 2. / (self.action_space[agent_id].nvec[1] - 1.) - 1.
        # norm_act[2] = action[2] * 2. / (self.action_space[agent_id].nvec[2] - 1.) - 1.
        # norm_act[3] = action[3] * 0.5 / (self.action_space[agent_id].nvec[3] - 1.) + 0.4

        return baseline_action

    def get_reward(self, env, agent_id, info: dict = ...) -> Tuple[float, dict]:
        if env.agents[agent_id].is_alive:
            return super().get_reward(env, agent_id, info=info)
        else:
            return 0.0, info

class Baseline2v2Agent:
    def __init__(self):
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.actor.eval()
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))  # hard code

    def normalize_action(self, action):
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.   # 0~40 => -1~1
        norm_act[1] = action[1] / 20 - 1.   # 0~40 => -1~1
        norm_act[2] = action[2] / 20 - 1.   # 0~40 => -1~1
        norm_act[3] = action[3] / 58 + 0.4  # 0~29 => 0.4~0.9
        return norm_act

    def get_action(self, sim: AircraftSimulator):
        # get single control baseline observation
        def get_delta_heading(ego_feature, enm_feature):
            ego_x, ego_y, ego_vx, ego_vy = ego_feature
            ego_v = np.linalg.norm([ego_vx, ego_vy])
            enm_x, enm_y, enm_vx, enm_vy = enm_feature
            delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
            R = np.linalg.norm([delta_x, delta_y])

            proj_dist = delta_x * ego_vx + delta_y * ego_vy
            ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

            side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
            return ego_AO * side_flag

        ego_x, ego_y, ego_z = sim.get_position()
        ego_vx, ego_vy, ego_vz = sim.get_velocity()
        enm_x, enm_y, enm_z = sim.enemies[0].get_position()
        enm_vx, enm_vy, enm_vz = sim.enemies[0].get_velocity()

        ego_feature = np.array([ego_x, ego_y, ego_vx, ego_vy])
        enm_feature = np.array([enm_x, enm_y, enm_vx, enm_vy])
        ego_AO = get_delta_heading(ego_feature, enm_feature)

        observation = np.zeros(12)
        observation[0] = (enm_z - ego_z) / 1000
        observation[1] = in_range_rad(ego_AO)
        # maintain same speed as enemy
        observation[2] = (sim.enemies[0].get_property_value(c.velocities_u_mps)
                          - sim.get_property_value(c.velocities_u_mps)) / 340
        observation[3] = ego_z / 5000
        observation[4] = np.sin(sim.get_rpy()[0])
        observation[5] = np.cos(sim.get_rpy()[0])
        observation[6] = np.sin(sim.get_rpy()[1])
        observation[7] = np.cos(sim.get_rpy()[1])
        observation[8] = sim.get_property_value(c.velocities_u_mps) / 340
        observation[9] = sim.get_property_value(c.velocities_v_mps) / 340
        observation[10] = sim.get_property_value(c.velocities_w_mps) / 340
        observation[11] = sim.get_property_value(c.velocities_vc_mps) / 340
        observation = np.expand_dims(observation, axis=0)   # dim: (1,12)

        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return self.normalize_action(action)
