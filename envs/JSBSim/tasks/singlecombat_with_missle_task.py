from collections import deque
import torch
import numpy as np
from gym import spaces

from .singlecombat_task import SingleCombatTask
from ..reward_functions import AltitudeReward, MissileAttackReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R, get_root_dir


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
        self.bloods, self.remaining_missiles, self.lock_duration = None, None, None

    def load_observation_space(self):
        self.observation_space = [spaces.Box(low=-10, high=10., shape=(25,)) for _ in range(self.num_agents)]

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]

    def normalize_observation(self, env, observations):
        """Convert simulation states into the format of observation_space
        """
        def _normalize(agent_id):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_aircrafts

            ego_obs_list, enm_obs_list = np.array(observations[ego_idx]), np.array(observations[enm_idx])
            # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
            ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
            enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
            ego_feature = np.array([*(ego_cur_ned/1000), *(ego_obs_list[6:9]/340)])
            enm_feature = np.array([*(enm_cur_ned/1000), *(enm_obs_list[6:9]/340)])
            observation = np.zeros(25)
            # (1) ego info normalization
            observation[0] = ego_obs_list[2] / 5000             #  0. ego altitude  (unit: 5km)
            observation[1] = np.linalg.norm(ego_feature[3:])    #  1. ego_v         (unit: mh)
            observation[2] = ego_obs_list[8] / 340              #  2. ego_v_down    (unit: mh)
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
            observation[12] = ego_AO                            # 12. ego_enm_AO        (unit: rad)
            observation[13] = ego_TA                            # 13. ego_enm_TA        (unit: rad)
            observation[14] = side_flag                         # 14. enm_delta_heading: 1 or 0 or -1
            observation[15] = enm_feature[2] / 5                # 15. enm_altitude  (unit: 5km)
            observation[16] = np.linalg.norm(enm_feature[3:])   # 16. enm_v         (unit: mh)
            observation[17] = enm_feature[5]                    # 17. enm_v_down    (unit: mh)
            # (3) missile info
            enm_missile_sim = self.check_missile_warning(env, agent_id)
            if enm_missile_sim is not None:
                enm_missile_feature = np.array([*(enm_missile_sim.get_position()/1000), *(enm_missile_sim.get_velocity()/340)])
                ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_missile_feature, return_side=True)
                observation[18] = R / 10                                    # 11. relative distance (unit: 10km)
                observation[19] = ego_AO                                    # 12. ego_missile_AO        (unit: rad)
                observation[20] = ego_TA                                    # 13. ego_missile_TA        (unit: rad)
                observation[21] = side_flag                                 # 14. missile_delta_heading: 1 or 0 or -1
                observation[22] = enm_missile_feature[2] / 5                # 15. missile_altitude  (unit: 5km)
                observation[23] = np.linalg.norm(enm_missile_feature[3:])   # 16. missile_v         (unit: mh)
                observation[24] = enm_missile_feature[5]                    # 17. missile_v_down    (unit: mh)
            return observation

        norm_obs = np.zeros((self.num_aircrafts, 25))
        for agent_id in range(self.num_aircrafts):
            norm_obs[agent_id] = _normalize(agent_id)
        return norm_obs

    def normalize_action(self, env, actions):
        return super().normalize_action(env, actions)

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self.bloods = [100 for _ in range(self.num_aircrafts)]
        self.remaining_missiles = [env.aircraft_configs[uid].get("missile", 0) for uid in env.aircraft_configs.keys()]
        self.lock_duration = [deque(maxlen=int(1 / env.time_interval)) for _ in range(self.num_aircrafts)]
        return super().reset(env)

    def step(self, env, action):
        for agent_id in range(self.num_aircrafts):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_aircrafts
            ego_uid, enm_uid = list(env.jsbsims.keys())[ego_idx], list(env.jsbsims.keys())[enm_idx]
            # [Rule-based missile launch]
            max_attack_angle = 22.5
            max_attack_distance = 6000
            target = env.jsbsims[enm_uid].get_position() - env.jsbsims[ego_uid].get_position()
            heading = env.jsbsims[ego_uid].get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) \
                / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[ego_idx].append(attack_angle < max_attack_angle)
            shoot_flag = np.sum(self.lock_duration[ego_idx]) >= self.lock_duration[ego_idx].maxlen \
                and distance <= max_attack_distance and self.remaining_missiles[ego_idx] > 0
            if shoot_flag:
                new_missile_uid = hex(int(env.jsbsims[ego_uid].uid, 16) + 1).lstrip("0x").upper()
                env.other_sims[new_missile_uid] = MissileSimulator.create(
                    parent=env.jsbsims[ego_uid],
                    target=env.jsbsims[enm_uid],
                    uid=new_missile_uid)
                self.remaining_missiles[ego_idx] -= 1
                print("Aircraft[{}] launched Missile[{}] --> target at Aircraft[{}]".format(
                    env.jsbsims[ego_uid].uid, new_missile_uid, env.jsbsims[enm_uid].uid
                ))
        active_sim_keys = list(env.other_sims.keys())
        for key in active_sim_keys:
            sim = env.other_sims[key]
            if isinstance(sim, MissileSimulator):
                if sim.is_success:
                    self.bloods[list(env.jsbsims.keys()).index(sim.target_aircraft.uid)] = 0
                elif sim.is_deleted:
                    env.other_sims.pop(sim.uid)

    def check_missile_warning(self, env, agent_id) -> MissileSimulator:
        ego_uid = list(env.jsbsims.keys())[agent_id]
        for sim in env.other_sims.values():
            if isinstance(sim, MissileSimulator):
                if sim.target_aircraft.uid == ego_uid:
                    return sim
        return None

    def get_termination(self, env, agent_id, info={}):
        done = False
        for condition in self.termination_conditions:
            d, _, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            # force terminate if time is out
            if d and isinstance(condition, Timeout):
                return True, info
            # otherwise(ego crash) wait until there is no ego-missile alive
            elif done:
                for sim in env.other_sims.values():
                    if isinstance(sim, MissileSimulator) and sim.is_alive and sim.target_aircraft.is_alive and \
                        sim._parent_uid == env.jsbsims[list(env.jsbsims.keys())[agent_id]].uid:
                        return False, info
        return done, info


class SingleCombatWithMissileHierarchicalTask(SingleCombatWithMissileTask):

    def __init__(self, config: str):
        super().__init__(config)
        self.norm_delta_altitude = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
        self.norm_delta_heading = [-90, 0, 90]
        assert len(self.norm_delta_altitude) == self.action_space[0].nvec[0]
        assert len(self.norm_delta_heading) == self.action_space[0].nvec[1]
        # low-level policy: Input(18) Output(4)
        self.lowlevel_policy = torch.load((get_root_dir() + '/model/singlecontrol_baseline.pth'))
        self.lowlevel_policy.eval()
        self._rnn_states = np.zeros((self.num_agents, 1, 1, 128))

    def load_action_space(self):
        self.action_space = [spaces.MultiDiscrete([7, 3]) for _ in range(self.num_agents)]

    def normalize_action(self, env, actions):
        """Convert high-level action into low value.
        """
        def _convert(action, observation, rnn_states):
            input_obs = np.zeros(8)
            input_obs[0] = self.norm_delta_altitude[action[0]]                  # 0. ego delta altitude  (unit: 1km)
            input_obs[1] = self.norm_delta_heading[action[1]] / 180 * np.pi     # 1. ego delta heading   (unit rad)
            input_obs[2] = observation[3]           # 2. ego_roll       (unit: rad)
            input_obs[3] = observation[4]           # 3. ego_pitch      (unit: rad)
            input_obs[4] = observation[6] / 340     # 4. ego_v_north   (unit: mh)
            input_obs[5] = observation[7] / 340     # 5. ego_v_east    (unit: mh)
            input_obs[6] = observation[8] / 340     # 6. ego_v_down    (unit: mh)
            input_obs[7] = observation[9] / 340     # 7. ego_vc        (unit: mh)
            input_obs = np.expand_dims(input_obs, axis=0)
            _action, _, _rnn_states = self.lowlevel_policy(input_obs, rnn_states, deterministic=True)
            action = _action.detach().cpu().numpy()
            rnn_states = _rnn_states.detach().cpu().numpy()
            return action, rnn_states

        def _normalize(action):
            action_norm = np.zeros(4)
            action_norm[0] = action[0] / 20 - 1.
            action_norm[1] = action[1] / 20 - 1.
            action_norm[2] = action[2] / 20 - 1.
            action_norm[3] = action[3] * 0.5 / 29 + 0.4
            return action_norm

        observations = env.get_observation(normalize=False)
        low_level_actions = np.zeros((self.num_agents, 4))
        for agent_id in range(self.num_agents):
            low_level_actions[agent_id], self._rnn_states[agent_id] = \
                _convert(actions[agent_id], observations[agent_id], self._rnn_states[agent_id])

        norm_act = np.zeros((self.num_aircrafts, 4))
        if self.use_baseline:
            norm_act[0] = _normalize(low_level_actions[0])
            norm_act[1] = _normalize(self.baseline_agent.get_action(env, self))
        else:
            for agent_id in range(self.num_aircrafts):
                norm_act[agent_id] = _normalize(low_level_actions[agent_id])
        return norm_act

    def reset(self, env):
        self._rnn_states = np.zeros((self.num_agents, 1, 1, 128))
        return super().reset(env)
