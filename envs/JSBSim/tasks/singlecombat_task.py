import numpy as np
from gym import spaces
import torch
from torch._C import device
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout
from ..utils.utils import in_range_rad, lonlat2dis, get_AO_TA_R

class SingleCombatTask(BaseTask):
    def __init__(self, config):
        self.config = config
        self.num_fighters = getattr(self.config, 'num_fighters', 2)
        assert self.num_fighters == 2, 'Only support one-to-one fighter combat!'
        self.bloods = [100 for _ in range(self.num_fighters)]
        self.use_baseline = getattr(self.config, 'use_baseline', False)
        self.num_agents = self.num_fighters - self.use_baseline  # output obs/act space
        self.which_baseline = getattr(self.config, 'which_baseline', 0)
        self.baseline_model_path = getattr(self.config, 'baseline_model_path', None)

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
        if self.use_baseline:
            if self.which_baseline == 'control':
                self.baseline_agent = SingleControlAgent(self.baseline_model_path)
            elif self.which_baseline == 'straight':
                self.baseline_agent = StraightFlyAgent()
            elif self.which_baseline == 'straight_continuous':
                self.baseline_agent = StraightFlyContinuousAgent()
            

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
        self.observation_space = [spaces.Box(low=-10, high=10., shape=(18,)) for _ in range(self.num_agents)]

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]

    def normalize_observation(self, env, observations):
        """Convert simulation states into the format of observation_space
        """
        def _normalize(agent_id):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_fighters
            ego_obs_list, enm_obs_list = observations[ego_idx], observations[enm_idx]
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

        norm_obs = np.zeros((self.num_fighters, 18))
        for agent_id in range(self.num_fighters):
            norm_obs[agent_id] = _normalize(agent_id)
        return norm_obs

    def normalize_action(self, env, actions):
        """Convert discrete action index into continuous value.
        """
        def _normalize(action):
            action_norm = np.zeros(4)
            action_norm[0] = action[0] * 2. / (self.action_space[0].nvec[0] - 1.) - 1.
            action_norm[1] = action[1] * 2. / (self.action_space[0].nvec[1] - 1.) - 1.
            action_norm[2] = action[2] * 2. / (self.action_space[0].nvec[2] - 1.) - 1.
            action_norm[3] = action[3] * 0.5 / (self.action_space[0].nvec[3] - 1.) + 0.4
            return action_norm

        norm_act = np.zeros((self.num_fighters, 4))
        for agent_id in range(self.num_fighters):
            norm_act[agent_id] = _normalize(actions[agent_id])

        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.

        Must call it after `env.get_observation()`
        """
        if self.use_baseline:
            self.baseline_agent.reset()
        return super().reset(env)

    def get_reward(self, env, agent_id, info={}):
        """
        Must call it after `env.get_observation()`
        """
        return super().get_reward(env, agent_id, info)

    def get_termination(self, env, agent_id, info={}):
        return super().get_termination(env, agent_id, info)


class SingleCombatContinuousTask(SingleCombatTask):
    '''
    Combat task with continuous action space
    '''
    def __init__(self, config):
        super().__init__(config)
    
    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.Box(
                                        low=np.array([-1.0, -1.0, -1.0, 0.4]), 
                                        high=np.array([1.0, 1.0, 1.0, 0.9])
                                         ) for _ in range(self.num_agents)]

    def normalize_action(self, env, actions: list):
        """Clip continuous value into proper value.
        """
        def _normalize(action):
            return np.clip(action, [-1.0, -1.0, -1.0, 0.4], [1.0, 1.0, 1.0, 0.9])

        norm_act = np.zeros((self.num_fighters, 4))
        for agent_id in range(self.num_fighters):
            norm_act[agent_id] = _normalize(actions[agent_id])
        return norm_act


import torch
class StraightFlyAgent:
    def __init__(self):
        pass

    def get_action(self, env, task):
        return np.array([20, 18.6, 20, 0])

    def reset(self):
        pass


class StraightFlyContinuousAgent:
    def __init__(self) -> None:
        pass

    def get_action(self, env, task):
        return np.array([0., -0.07 ,0., 0.4])
    
    def reset(self):
        pass


class SingleControlAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.restore()
        self.prep_rollout()
        self.reset()
    
    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128)) # hard code

    def get_action(self, env, task):
        # get single control baseline observation
        def get_delta_heading(ego_feature, enm_feature):
            ego_x, ego_y, ego_vx, ego_vy = ego_feature
            ego_v = np.linalg.norm([ego_vx, ego_vy])
            enm_x, enm_y, enm_vx, enm_vy = enm_feature
            enm_v = np.linalg.norm([enm_vx, enm_vy])
            delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
            R = np.linalg.norm([delta_x, delta_y])

            proj_dist = delta_x * ego_vx + delta_y * ego_vy
            ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

            side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
            return - ego_AO * side_flag

        ego_id, enm_id= 1, 0
        ego_obs_list = env.sims[1].get_property_values(task.state_var)
        enm_obs_list = env.sims[0].get_property_values(task.state_var)

        ego_cur_east, ego_cur_north = lonlat2dis(ego_obs_list[0], ego_obs_list[1], env.init_longitude, env.init_latitude)
        enm_cur_east, enm_cur_north = lonlat2dis(enm_obs_list[0], enm_obs_list[1], env.init_longitude, env.init_latitude)
        ego_feature = np.array([
            ego_cur_east / 1000, ego_cur_north / 1000, 
            ego_obs_list[7] * 0.304 / 340, ego_obs_list[6] * 0.304 / 340
        ])
        enm_feature = np.array([
            enm_cur_east / 1000, enm_cur_north / 1000,
            enm_obs_list[7] * 0.304 / 340, enm_obs_list[6] * 0.304 / 340
        ])
        ego_AO = get_delta_heading(ego_feature, enm_feature)

        observation = np.zeros(8)
        observation[0] = (enm_obs_list[2]-ego_obs_list[2]) * 0.304 / 1000         #  0. ego delta altitude  (unit: 1km)
        observation[1] = in_range_rad(ego_AO)               #  1. ego delta heading   (unit rad)
        observation[2] = ego_obs_list[3]                    #  2. ego_roll    (unit: rad)
        observation[3] = ego_obs_list[4]                    #  3. ego_pitch   (unit: rad)
        observation[4] = ego_obs_list[6] * 0.304 / 340      #  4. ego_v_north        (unit: mh)
        observation[5] = ego_obs_list[7] * 0.304 / 340      #  5. ego_v_east        (unit: mh)
        observation[6] = ego_obs_list[8] * 0.304 / 340      #  6. ego_v_down        (unit: mh)
        observation[7] = ego_obs_list[9] * 0.304 / 340      #  7. ego_vc        (unit: mh)
        observation = np.expand_dims(observation, axis=0)    # dim: (1,8)

        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return  action

    def restore(self):
        self.actor = torch.load(str(self.model_path))

    def prep_rollout(self):
        self.actor.eval()