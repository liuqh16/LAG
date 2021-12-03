import torch
import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..utils.utils import in_range_rad, get_AO_TA_R, LLA2NEU, get_root_dir


class SingleCombatTask(BaseTask):
    def __init__(self, config):
        self.config = config
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        if self.use_baseline:
            self.load_policy(self.config.baseline_type)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    @property
    def num_agents(self) -> int:
        return 1

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

    def normalize_obs(self, env, obs, enemy_obs):
        """Combine both aircrafts' state to generate actual observation
        """
        norm_obs = np.zeros(18)
        ego_obs_list, enm_obs_list = np.array(obs), np.array(enemy_obs)
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

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        if self.policy is not None:
            self.policy.reset()
        return super().reset(env)

    def load_policy(self, policy: str):
        if isinstance(policy, str):
            if policy == "control":
                self.policy = SingleControlAgent()
            elif policy == "straight":
                self.policy = StraightFlyAgent()
            else:
                assert ValueError(f"Unknown policy: {policy}")
        else:
            assert isinstance(policy, torch.nn.Module)
            self.policy = HistoryAgent(policy)

    def rollout(self, env, sim):
        return self.policy.get_action(env, self, sim)


def load_agent(name):
    if name == 'control':
        return SingleControlAgent()
    elif name == 'straight':
        return StraightFlyAgent()
    else:
        raise NotImplementedError


class SingleControlAgent:
    def __init__(self):
        self.actor = torch.load(get_root_dir() + '/model/singlecontrol_baseline.pth').cpu()
        self.actor.eval()
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, self.actor.recurrent_hidden_layers, self.actor.recurrent_hidden_size))

    def _calculate_cmd(self, env, task, sim):

        ego_sim, enm_sim = sim, sim.enemies[0]
        ego_x, ego_y, ego_z = ego_sim.get_position()
        ego_vx, ego_vy, ego_vz = ego_sim.get_velocity()
        enm_x, enm_y, enm_z = enm_sim.get_position()

        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])

        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        ego_AO *= np.sign(np.cross([delta_x, delta_y], [ego_vx, ego_vy]))

        delta_altitude = (enm_z - ego_z) / 1000
        delta_heading = in_range_rad(ego_AO)
        return delta_altitude, delta_heading

    def get_action(self, env, task, sim):

        delta_altitude, delta_heading = self._calculate_cmd(env, task, sim)
        ego_roll, ego_pitch = sim.get_rpy()[0:2]
        ego_vn, ego_ve, ego_vu = sim.get_velocity()

        observation = np.zeros(8)
        observation[0] = delta_altitude     # 0. ego delta altitude  (unit: 1km)
        observation[1] = delta_heading      # 1. ego delta heading   (unit rad)
        observation[2] = ego_roll           # 2. ego_roll      (unit: rad)
        observation[3] = ego_pitch          # 3. ego_pitch     (unit: rad)
        observation[4] = ego_vn / 340       # 4. ego_v_north   (unit: mh)
        observation[5] = ego_ve / 340       # 5. ego_v_east    (unit: mh)
        observation[6] = ego_vu / 340       # 6. ego_v_up    (unit: mh)
        observation[7] = sim.get_property_value(c.velocities_vc_mps) / 340
        observation = np.expand_dims(observation, axis=0)   # dim: (1,8)

        action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        return action.detach().cpu().numpy().squeeze()


class StraightFlyAgent(SingleControlAgent):

    def _calculate_cmd(self, env, task, sim):
        return 0, 0


class HistoryAgent(SingleControlAgent):
    def __init__(self, policy: torch.nn.Module):
        self.actor = policy.cpu()
        self.actor.eval()

    def get_action(self, env, task, sim):
        ego_obs = sim.get_property_values(task.state_var)
        enm_obs = sim.ememies[0].get_property_values(task.state_var)
        norm_obs = task.normalize_obs(env, ego_obs, enm_obs)
        norm_obs = np.expand_dims(norm_obs, axis=0)
        action, _, self.rnn_states = self.actor(norm_obs, self.rnn_states, deterministic=True)
        return action.detach().cpu().numpy().squeeze()
