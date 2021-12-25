from collections import deque
import torch
import numpy as np
from gym import spaces

from .singlecombat_task import SingleCombatTask, BaselineActor
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward
from ..core.simulatior import MissileSimulator
from ..utils.utils import LLA2NEU, get2d_AO_TA_R, get_root_dir


class SingleCombatWithMissileTask(SingleCombatTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def get_obs(self, env, agent_id):
        """Convert simulation states into the format of observation_space

        (1) ego info
            0. ego altitude           (unit: 5km)
            1. ego_roll_sin
            2. ego_roll_cos
            3. ego_pitch_sin
            4. ego_pitch_cos
            5. ego v_body_x           (unit: mh)
            6. ego v_body_y           (unit: mh)
            7. ego v_body_z           (unit: mh)
            8. ego_vc                 (unit: mh)
        (2) relative enm info
            9. delta_v_body_x        (unit: mh)
            10. delta_altitude        (unit: km)
            11. ego_AO                (unit: rad) [0, pi]
            12. ego_TA                (unit: rad) [0, pi]
            13. relative distance     (unit: 10km)
            14. side_flag             1 or 0 or -1
        (3) relative missile info
            15. delta_v_body_x
            16. delta altitude
            17. ego_AO
            18. ego_TA
            19. relative distance
            20. side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*(ego_cur_ned / 1000), *(ego_obs_list[6:9] / 340)])
        enm_feature = np.array([*(enm_cur_ned / 1000), *(enm_obs_list[6:9] / 340)])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = self.check_missile_warning(env, agent_id)
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get2d_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self.remaining_missiles = dict([(agent_id, env.config.aircraft_configs[agent_id].get("missile", 0)) for agent_id in env.agents.keys()])
        self.lock_duration = dict([(agent_id, deque(maxlen=int(1 / env.time_interval))) for agent_id in env.agents.keys()])
        self.max_attack_distance = env.np_random.uniform(5000, 10000)
        return super().reset(env)

    def step(self, env):
        for agent_id, agent in env.agents.items():
            # [Rule-based missile launch]
            max_attack_angle = 22.5
            max_attack_distance = self.max_attack_distance
            target = agent.enemies[0].get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < max_attack_angle)
            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= max_attack_distance and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1

    def check_missile_warning(self, env, agent_id) -> MissileSimulator:
        for missile in env.agents[agent_id].under_missiles:
            if missile.is_alive:
                return missile
        return None


class SingleCombatWithMissileHierarchicalTask(SingleCombatWithMissileTask):

    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cpu')))
        self.lowlevel_policy.eval()
        # self.norm_delta_altitude = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
        self.norm_delta_heading = np.array([-np.pi/6, -np.pi/12, 0, np.pi / 12, np.pi / 6])

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([5])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            input_obs[0] = raw_obs[10]
            input_obs[1] = self.norm_delta_heading[action]
            input_obs[2] = raw_obs[9]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4
            return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)


class SingleCombatShootHierarchicalTask(SingleCombatWithMissileHierarchicalTask):
    def __init__(self, config: str):
        super().__init__(config)
        self.shoot_action = None
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            EventDrivenReward(self.config)
        ]
    
    def reset(self, env):
        super().reset(env)
        self.max_attack_distance = 8000

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([5, 2])

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        if agent_id in env.ego_ids:
            self.shoot_action = action[1]
        return super().normalize_action(env, agent_id, action[0])

    def step(self, env):
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch]
            target = agent.enemies[0].get_position() - agent.get_position()
            distance = np.linalg.norm(target)
            max_attack_distance = self.max_attack_distance
            shoot_flag = agent.is_alive and self.shoot_action \
                and distance <= max_attack_distance and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1