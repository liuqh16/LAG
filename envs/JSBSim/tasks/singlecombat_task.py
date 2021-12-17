import torch
import numpy as np
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..reward_functions import AltitudeReward, PostureReward, CrashReward
from ..utils.utils import get2d_AO_TA_R, in_range_rad, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor


class SingleCombatTask(BaseTask):
    def __init__(self, config):
        self.config = config
        self.num_aircrafts = len(getattr(self.config, 'aircraft_configs', {}).keys())
        assert self.num_aircrafts == 2, 'Only support one-to-one air combat!'
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        if self.use_baseline:
            self.baseline_agent = self.load_agent(self.config.baseline_type, agent_id=1)

        self.reward_functions = [
            AltitudeReward(self.config),
            PostureReward(self.config),
            CrashReward(self.config)
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
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
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
        self.observation_space = spaces.Box(low=-10, high=10., shape=(15,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        (1) ego info
            0. ego altitude         (unit: 5km)
            1. ego_roll_sin
            2. ego_roll_cos
            3. ego_pitch_sin
            4. ego_pitch_cos
            5. ego v_body_x         (unit: mh)
            6. ego v_body_y         (unit: mh)
            7. ego v_body_z         (unit: mh)
            8. ego_vc               (unit: mh)
        (2) relative info
            9. delta_v_body_x       (unit: mh)
            10. delta_altitude      (unit: km)
            11. ego_AO              (unit: rad) [0, pi]
            12. ego_TA              (unit: rad) [0, pi]
            13. relative distance   (unit: 10km)
            14. side_flag           1 or 0 or -1
        """
        norm_obs = np.zeros(15)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000            # 0. ego altitude   (unit: 5km)
        norm_obs[1] = np.sin(ego_obs_list[3])           # 1. ego_roll_sin
        norm_obs[2] = np.cos(ego_obs_list[3])           # 2. ego_roll_cos
        norm_obs[3] = np.sin(ego_obs_list[4])           # 3. ego_pitch_sin
        norm_obs[4] = np.cos(ego_obs_list[4])           # 4. ego_pitch_cos
        norm_obs[5] = ego_obs_list[9] / 340             # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = ego_obs_list[10] / 340            # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = ego_obs_list[11] / 340            # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = ego_obs_list[12] / 340            # 8. ego vc   (unit: mh)
        # (2) relative info w.r.t enm state
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
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

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        if self.use_baseline:
            self.baseline_agent.reset()
        return super().reset(env)

    def load_agent(self, name, agent_id):
        if name == 'control':
            return SingleControlAgent(agent_id)
        elif name == 'straight':
            return StraightFlyAgent()
        else:
            raise NotImplementedError


class StraightFlyAgent:
    def get_action(self, env, task):
        return np.array([20, 18.6, 20, 0])

    def reset(self):
        pass


class SingleControlAgent:
    def __init__(self, agent_id=1):
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path))
        self.actor.eval()
        self.reset()
        self.agent_id = agent_id

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))  # hard code

    def get_action(self, env, task):
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

        ego_uid, enm_uid = list(env.jsbsims.keys())[self.agent_id], list(env.jsbsims.keys())[(self.agent_id + 1) % 2]
        ego_x, ego_y, ego_z = env.jsbsims[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.jsbsims[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.jsbsims[enm_uid].get_position()
        enm_vx, enm_vy, enm_vz = env.jsbsims[enm_uid].get_velocity()

        ego_feature = np.array([ego_x, ego_y, ego_vx, ego_vy])
        enm_feature = np.array([enm_x, enm_y, enm_vx, enm_vy])
        ego_AO = get_delta_heading(ego_feature, enm_feature)

        observation = np.zeros(12)
        observation[0] = (enm_z - ego_z) / 1000
        observation[1] = in_range_rad(ego_AO)
        observation[2] = (243 - env.jsbsims[ego_uid].get_property_value(c.velocities_u_mps)) / 340
        observation[3] = ego_z / 5000
        observation[4] = np.sin(env.jsbsims[ego_uid].get_rpy()[0])
        observation[5] = np.cos(env.jsbsims[ego_uid].get_rpy()[0])
        observation[6] = np.sin(env.jsbsims[ego_uid].get_rpy()[1])
        observation[7] = np.cos(env.jsbsims[ego_uid].get_rpy()[1])
        observation[8] = env.jsbsims[ego_uid].get_property_value(c.velocities_u_mps) / 340
        observation[9] = env.jsbsims[ego_uid].get_property_value(c.velocities_v_mps) / 340
        observation[10] = env.jsbsims[ego_uid].get_property_value(c.velocities_w_mps) / 340
        observation[11] = env.jsbsims[ego_uid].get_property_value(c.velocities_vc_mps) / 340
        observation = np.expand_dims(observation, axis=0)   # dim: (1,12)

        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action
