from abc import ABC
import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_AO_TA_R,get_root_dir
from envs.JSBSim.tasks import SingleCombatWithMissileTask
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from typing import Literal
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = get_root_dir() + '/model/actor.pth'
        self.actor = torch.load(str(self.model_path))
        self.actor.eval()
        self.agent_id = agent_id
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        uid = list(env.jsbsims.keys())[self.agent_id]
        obs = env.jsbsims[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = np.sin(obs[3])                 #  3. ego_roll_sin
        norm_obs[4] = np.cos(obs[3])                 #  4. ego_roll_cos
        norm_obs[5] = np.sin(obs[4])                 #  5. ego_pitch_sin
        norm_obs[6] = np.cos(obs[4])                 #  6. ego_pitch_cos
        norm_obs[7] = obs[5] / 340                   #  7. ego_v_x   (unit: mh)
        norm_obs[8] = obs[6] / 340                   #  8. ego_v_y    (unit: mh)
        norm_obs[9] = obs[7] / 340                   #  9. ego_v_z    (unit: mh)
        norm_obs[10] = obs[8] / 340                  #  10. ego_vc        (unit: mh)
        norm_obs[11] = obs[9] / 1000                 #  11. ego_altitude (unit: km)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action


class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, env, task):
        ego_uid, enm_uid = list(env.jsbsims.keys())[self.agent_id], list(env.jsbsims.keys())[(self.agent_id+1)%2] # NOTE: only adapt for 1v1
        ego_x, ego_y, ego_z = env.jsbsims[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.jsbsims[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.jsbsims[enm_uid].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.jsbsims[enm_uid].get_property_value(c.velocities_u_mps) - env.jsbsims[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n', 'o', '0']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False # start turn when missile is detected, if set true
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'o':
            self.target_heading_list = [np.pi/2, np.pi, 3*np.pi/2, 0, 0]
        elif maneuver == '0':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi, -np.pi/2, -np.pi/2, 0]
        elif maneuver == 'triangle':
            self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]*10
        self.target_altitude_list = [8000, 7000, 7500, 5500, 6000, 6000]
        self.target_velocity_list = [340, 300, 150, 200, 243, 243]

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / env.time_interval
        uid = list(env.jsbsims.keys())[self.agent_id]
        cur_heading = env.jsbsims[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
        if not self.dodge_missile or task.check_missile_warning(env, self.agent_id) is None:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
        # delta altitude
        delta_altitude = self.target_altitude_list[i] - env.jsbsims[uid].get_property_value(c.position_h_sl_m)
        # delta heading
        delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
        # delta velocities
        delta_velocity = self.target_velocity_list[i] - env.jsbsims[uid].get_property_value(c.velocities_u_mps)
        self.step += 1
        return np.array([delta_altitude, delta_heading, delta_velocity])


def test_maneuver():
    env = SingleCombatEnv(config_name='1v1/Missile/test/opposite')
    env.reset()
    env.render()
    agent0 = ManeuverAgent(agent_id=0, maneuver='triangle')
    agent1 = PursueAgent(agent_id=1)
    reward_list = []
    while True:
        actions = [agent0.get_action(env, env.task), agent1.get_action(env, env.task)]
        next_obs, reward, done, info = env.step(actions)
        env.render()
        reward_list.append(reward[0])
        if np.array(done).all():
            print(info)
            break
    # plt.plot(reward_list)
    # plt.savefig('rewards.png')

if __name__ == '__main__':
    test_maneuver()