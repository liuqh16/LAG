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

class ManeuverAgent:
    def __init__(self, agent_id: Literal[0, 1], maneuver: Literal['l', 'r', 'n', 'o', '0']):
        self.model_path = get_root_dir() + '/model/singlecontrol_baseline.pth'
        self.ego_idx = agent_id
        self.restore()
        self.prep_rollout()
        self.step = 0
        self.seconds_per_turn = 7  # hyperparameter
        self.init_heading = None
        if maneuver == 'l':
            self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]*10
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'o':
            self.target_heading_list = [np.pi/2, np.pi, 3*np.pi/2, 0, 0]
        elif maneuver == '0':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi, -np.pi/2, -np.pi/2, 0]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))
        self.step = 0

    def get_action(self, env: SingleCombatEnv, task: SingleCombatWithMissileTask):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.seconds_per_turn / env.time_interval
        ego_uid = list(env.jsbsims.keys())[self.ego_idx]
        ego_obs_list = env.jsbsims[ego_uid].get_property_values(task.state_var)
        if self.init_heading is None:
            self.init_heading = ego_obs_list[5]
        delta_heading = 0

        # missile = task.check_missile_warning(env, self.ego_idx)
        # if missile is None:
        #     delta_heading = (self.init_heading - ego_obs_list[5])  # ego_obs_list[5] is ego's heading
        # else:
        for i, interval in enumerate(step_list):
            if self.step <= interval:
                break
        target_heading = self.target_heading_list[i]
        delta_heading = (self.init_heading + target_heading - ego_obs_list[5])
        self.step += 1

        observation = np.zeros(8)
        observation[0] = 0                              #  0. ego delta altitude  (unit: 1km)
        observation[1] = in_range_rad(delta_heading)    #  1. ego delta heading   (unit rad)
        observation[2] = ego_obs_list[3]                #  2. ego_roll      (unit: rad)
        observation[3] = ego_obs_list[4]                #  3. ego_pitch     (unit: rad)
        observation[4] = ego_obs_list[6] / 340          #  4. ego_v_north   (unit: mh)
        observation[5] = ego_obs_list[7] / 340          #  5. ego_v_east    (unit: mh)
        observation[6] = ego_obs_list[8] / 340          #  6. ego_v_down    (unit: mh)
        observation[7] = ego_obs_list[9] / 340          #  7. ego_vc        (unit: mh)
        observation = np.expand_dims(observation, axis=0)   # dim: (1,8)
        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action

    def restore(self):
        self.actor = torch.load(str(self.model_path))

    def prep_rollout(self):
        self.actor.eval()

class BaselineAgent:
    def __init__(self, ego_id):
        self.model_path = get_root_dir() + '/model/singlecontrol_baseline.pth'
        self.actor = torch.load(str(self.model_path))
        self.actor.eval()
        self.ego_id = ego_id
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128)) # hard code

    def get_action(self, env, task=None):
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
            return ego_AO * side_flag

        ego_uid, enm_uid = list(env.jsbsims.keys())[self.ego_id], list(env.jsbsims.keys())[(self.ego_id+1)%2]
        ego_x, ego_y, ego_z = env.jsbsims[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.jsbsims[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.jsbsims[enm_uid].get_position()
        enm_vx, enm_vy, enm_vz = env.jsbsims[enm_uid].get_velocity()

        ego_feature = np.array([ego_x, ego_y, ego_vx, ego_vy])
        enm_feature = np.array([enm_x, enm_y, enm_vx, enm_vy])
        ego_AO = get_delta_heading(ego_feature, enm_feature)
        ego_obs_list = env.jsbsims[ego_uid].get_property_values(task.state_var)
        observation = np.zeros(8)
        observation[0] = 0            #  0. ego delta altitude  (unit: 1km)
        observation[1] = in_range_rad(ego_AO)               #  1. ego delta heading   (unit rad)
        observation[2] = ego_obs_list[3]                #  2. ego_roll      (unit: rad)
        observation[3] = ego_obs_list[4]                #  3. ego_pitch     (unit: rad)
        observation[4] = ego_obs_list[6] / 340          #  4. ego_v_north   (unit: mh)
        observation[5] = ego_obs_list[7] / 340          #  5. ego_v_east    (unit: mh)
        observation[6] = ego_obs_list[8] / 340          #  6. ego_v_down    (unit: mh)
        observation[7] = ego_obs_list[9] / 340          #  7. ego_vc        (unit: mh)
        observation = np.expand_dims(observation, axis=0)   # dim: (1,8)

        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action

class BaselineDeltaVAgent(BaselineAgent):
    def __init__(self, ego_id):
        super().__init__(ego_id)
        self.model_path = get_root_dir() + '/model/singlecontrol_baseline_delta_V.pth'
        self.actor = torch.load(str(self.model_path))
        self.actor.eval()
        self.ego_id = ego_id
        self.reset()

    def get_action(self, env, task=None):
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
            return ego_AO * side_flag

        ego_uid, enm_uid = list(env.jsbsims.keys())[self.ego_id], list(env.jsbsims.keys())[(self.ego_id+1)%2]
        ego_x, ego_y, ego_z = env.jsbsims[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.jsbsims[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.jsbsims[enm_uid].get_position()
        enm_vx, enm_vy, enm_vz = env.jsbsims[enm_uid].get_velocity()

        ego_feature = np.array([ego_x, ego_y, ego_vx, ego_vy])
        enm_feature = np.array([enm_x, enm_y, enm_vx, enm_vy])
        ego_AO = get_delta_heading(ego_feature, enm_feature)
        ego_obs_list = env.jsbsims[ego_uid].get_property_values(task.state_var)
        observation = np.zeros(11)
        observation[0] = (8000-ego_obs_list[2])/1000          #  0. ego delta altitude  (unit: 1km)
        observation[1] = in_range_rad((np.pi - ego_obs_list[5]))               #  1. ego delta heading   (unit rad)
        observation[2] = (340-env.jsbsims[ego_uid].get_property_value(c.velocities_u_mps))/340         #  2. ego delta velocities_u  (unit: mh)
        observation[3] = np.sin(ego_obs_list[2])            #  3. ego_roll_sin
        observation[4] = np.cos(ego_obs_list[2])            #  4. ego_roll_cos
        observation[5] = np.sin(ego_obs_list[3])            #  5. ego_pitch_sin
        observation[6] = np.cos(ego_obs_list[3])            #  6. ego_pitch_cos
        observation[7] = ego_obs_list[4]
        observation[8] = env.jsbsims[ego_uid].get_property_value(c.velocities_u_mps) / 340          #  4. ego_v_north   (unit: mh)
        observation[9] = env.jsbsims[ego_uid].get_property_value(c.velocities_v_mps) / 340          #  5. ego_v_east    (unit: mh)
        observation[10] = env.jsbsims[ego_uid].get_property_value(c.velocities_w_mps) / 340          #  6. ego_v_down    (unit: mh)
                 #  7. ego_vc        (unit: mh)
        observation = np.expand_dims(observation, axis=0)    # dim: (1,11)

        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action

def test_maneuver():
    env = SingleCombatEnv(config_name='1v1/Missile/test/opposite')
    env.reset()
    env.render()
    # agent0 = ManeuverAgent(agent_id=0, maneuver='r')
    agent0 = BaselineDeltaVAgent(0)
    # agent1 = BaselineAgent(1)
    agent1 = ManeuverAgent(1, maneuver='l')
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