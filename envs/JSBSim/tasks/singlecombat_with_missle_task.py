import numpy as np
from collections import deque
from gym import spaces

from .singlecombat_task import SingleCombatTask, BaseTask
from ..reward_functions import AltitudeReward, MissileAttackReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, SafeReturn, Timeout
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R


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
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(25,))

    def get_obs(self, env, agent_id):
        norm_obs = np.zeros(25)
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
        # (3) extra missile info
        enm_missile_sim = self.check_missile_warning(env, agent_id)
        if enm_missile_sim is not None:
            enm_missile_feature = np.array([*(enm_missile_sim.get_position() / 1000), *(enm_missile_sim.get_velocity() / 340)])
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_missile_feature, return_side=True)
            norm_obs[18] = R / 10                                    # 11. relative distance (unit: 10km)
            norm_obs[19] = ego_AO                                    # 12. ego_missile_AO        (unit: rad)
            norm_obs[20] = ego_TA                                    # 13. ego_missile_TA        (unit: rad)
            norm_obs[21] = side_flag                                 # 14. missile_delta_heading: 1 or 0 or -1
            norm_obs[22] = enm_missile_feature[2] / 5                # 15. missile_altitude  (unit: 5km)
            norm_obs[23] = np.linalg.norm(enm_missile_feature[3:])   # 16. missile_v         (unit: mh)
            norm_obs[24] = enm_missile_feature[5]                    # 17. missile_v_down    (unit: mh)
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self.bloods = dict([(agent_id, 100) for agent_id in self.agent_ids])
        self.remaining_missiles = dict([(agent_id, env.config.aircraft_configs[agent_id].get("missile", 0)) for agent_id in self.agent_ids])
        self.lock_duration = dict([(agent_id, deque(maxlen=int(1 / env.time_interval))) for agent_id in self.agent_ids])
        return super().reset(env)

    def step(self, env):
        for agent_id in self.agent_ids:
            # [Rule-based missile launch]
            max_attack_angle = 22.5
            max_attack_distance = 12000
            target = env.agents[agent_id].enemies[0].get_position() - env.agents[agent_id].get_position()
            heading = env.agents[agent_id].get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < max_attack_angle)
            shoot_flag = env.agents[agent_id].is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= max_attack_distance and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = env.agents[agent_id].uid + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=env.agents[agent_id], target=env.agents[agent_id].enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1

    def check_missile_warning(self, env, agent_id) -> MissileSimulator:
        for missile in env.agents[agent_id].under_missiles:
            if missile.is_alive:
                return missile
        return None
