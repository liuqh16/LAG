import numpy as np
from collections import OrderedDict
from gym import spaces
from .singlecombat_task import SingleCombatTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, MissileAttackReward,PostureReward, RelativeAltitudeReward, MissilePostureReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..utils.missile_utils import Missile3D
from ..utils.utils import NEU2LLA, LLA2NEU, get_AO_TA_R


class SingleCombatWithMissileTask(SingleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)            

        self.reward_functions = [
            MissileAttackReward(self.config),
            AltitudeReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            ShootDown(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]
        self.init_missile()

    def init_missile(self):
        self.bloods = [100 for _ in range(self.num_fighters)]
        self.missile_lists = [Missile3D() for _ in range(self.num_fighters)] # By default, both figher has 1 missile.

    def load_observation_space(self):
        self.observation_space = [spaces.Box(low=-10, high=10., shape=(18,)) for _ in range(self.num_agents)]
        # TODO: add extra obs_space for missile information here!

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]
        # TODO: if need to control missile launch mannully, add a new axes in self.action_space

    def normalize_observation(self, env, observations):
        """Convert simulation states into the format of observation_space
        """
        def _normalize(agent_id):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_fighters
            ego_obs_list, enm_obs_list = np.array(observations[ego_idx]), np.array(observations[enm_idx])
            # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
            ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.init_longitude, env.init_latitude)
            enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.init_longitude, env.init_latitude)
            ego_feature = np.array([*(ego_cur_ned / 1000), *(ego_obs_list[6:9] / 340)])
            enm_feature = np.array([*(enm_cur_ned / 1000), *(enm_obs_list[6:9] / 340)])
            observation = np.zeros(18)
            # (1) ego info normalization
            observation[0] = ego_obs_list[2] / 5000             #  0. ego altitude  (unit: 5km)
            observation[1] = np.linalg.norm(ego_feature[3:])    #  1. ego_v         (unit: mh)
            observation[2] = ego_obs_list[8]                    #  2. ego_v_down    (unit: mh)
            observation[3] = np.sin(ego_obs_list[3])            #  3. ego_roll_sin
            observation[4] = np.cos(ego_obs_list[3])            #  4. ego_roll_cos
            observation[5] = np.sin(ego_obs_list[4])            #  5. ego_pitch_sin
            observation[6] = np.cos(ego_obs_list[4])            #  6. ego_pitch_cos
            observation[7] = ego_obs_list[9] / 340              #  7. ego_vc        (unit: mh)
            observation[8] = ego_obs_list[10]                   #  8. ego_north_ng  (unit: 5G)
            observation[9] = ego_obs_list[11]                   #  9. ego_east_ng   (unit: 5G)
            observation[10] = ego_obs_list[12]                  # 10. ego_down_ng   (unit: 5G)
            # (2) relative info w.r.t enm state
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
            observation[11] = R / 10                            # 11. relative distance (unit: 10km)
            observation[12] = ego_AO                            # 12. ego_AO        (unit: rad)
            observation[13] = ego_TA                            # 13. ego_TA        (unit: rad)
            observation[14] = side_flag                         # 14. enm_delta_heading: 1 or 0 or -1
            observation[15] = enm_obs_list[2] / 5000            # 15. enm_altitude  (unit: 5km)
            observation[16] = np.linalg.norm(enm_feature[3:])   # 16. enm_v         (unit: mh)
            observation[17] = enm_obs_list[8]                   # 17. enm_v_down    (unit: mh)
            # (3) missile state
            if self.missile_lists[enm_idx].missile_info[0]['flying']:
                enm_missile_feature = self.missile_lists[enm_idx].missile_info[0]['current_state'] # get missile state TODO: if there are multi missiles
                enm_missile_feature = self.missile_lists[enm_idx].simulator.transfer2raw(enm_missile_feature) # transform coordinate system
                enm_missile_feature = np.array([*(enm_missile_feature[:3]/1000), *(enm_missile_feature[3:6]/340)]) # transform unit
                ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_missile_feature, return_side=True)
                observation[11] = R / 10                            # 11. relative distance (unit: 10km)
                observation[12] = ego_AO                            # 12. ego_AO        (unit: rad)
                observation[13] = ego_TA                            # 13. ego_TA        (unit: rad)
                observation[14] = side_flag                         # 14. enm_delta_heading: 1 or 0 or -1
                observation[15] = enm_missile_feature[2] / 5000            # 15. enm_altitude  (unit: 5km)
                observation[16] = np.linalg.norm(enm_missile_feature[3:])   # 16. enm_v         (unit: mh)
                observation[17] = enm_missile_feature[5]                   # 17. enm_v_down    (unit: mh)
            return observation

        norm_obs = np.zeros((self.num_fighters, 18))
        for agent_id in range(self.num_fighters):
            norm_obs[agent_id] = _normalize(agent_id)
        return norm_obs

    def normalize_action(self, env, actions):
        return super().normalize_action(env, actions)

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self.init_missile()
        return super().reset(env)

    def step(self, env, action):
        for agent_id in range(self.num_fighters):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_fighters
            ego_feature = np.hstack([env.sims[ego_idx].get_position(), env.sims[ego_idx].get_velocity()])
            enm_feature = np.hstack([env.sims[enm_idx].get_position(), env.sims[enm_idx].get_velocity()])
            flag_hit = self.missile_lists[ego_idx].make_step(ego_feature, enm_feature, env.current_step)
            self.bloods[enm_idx] *= 1. - flag_hit

    def get_termination(self, env, agent_id, info={}):
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success or s
            # force terminate if time is out
            if d and isinstance(condition, Timeout):
                break
            # otherwise(ego crash) wait until there is no ego-missile alive
            elif done and self.missile_lists[agent_id].check_no_missile_alive():
                break
        return done, info

    def get_missile_trajectory(self, env, agent_id):
        """
        render missile trajectory, if no missile, use default value instead
        """
        missile_render = []
        for i in range(self.missile_lists[agent_id].num_missile):
            missile_state = np.zeros((6,))
            if self.missile_lists[agent_id].missile_info[i]['launched']:
                missile_state[:3] = np.array(self.missile_lists[agent_id].missile_info[i]['current_state'][:3])
                missile_state[4:6] = np.array(self.missile_lists[agent_id].missile_info[i]['current_state'][4:6])[::-1]
            else:
                missile_state[:3] = env.sims[agent_id].get_position()
                missile_state[4:6] = env.sims[agent_id].get_rpy()[1:]
            # unit conversion (m, m, m) => (degree, degree, ft)
            missile_state[:3] = NEU2LLA(*missile_state[:3], env.init_longitude, env.init_latitude)
            missile_render.append(missile_state)
        return missile_render


class SingleCombatWithAvoidMissileTask(SingleCombatWithMissileTask):
    def init_missile(self):
        self.bloods = [100 for _ in range(self.num_fighters)]
        self.missile_lists = [Missile3D(allow_shoot=False), Missile3D(allow_shoot=True)] # By default, both figher has 1 missile.