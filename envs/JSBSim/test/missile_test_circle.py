# Deal with import error
import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from envs.JSBSim.envs import SingleControlEnv, SingleCombatEnv
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad
import torch 
import time
import numpy as np

STEPS_PER_SECOND = 12

class CircleAgent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.restore()
        self.prep_rollout()
        self.step = 0
        self.seconds_per_turn = 5  # hyperparameter
        self.step_list = np.array([1,2,3,4]) * self.seconds_per_turn * STEPS_PER_SECOND
        self.target_heading_list = [np.pi/2, np.pi, np.pi*1.5, 0]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128)) # hard code
        self.step = 0

    def get_action(self, env, task):
        ego_obs_list = env.sims[0].get_property_values(task.state_var)
        enm_obs_list = env.sims[1].get_property_values(task.state_var)
        delta_heading = 0

        if not task.missile_lists[1].missile_info[0]['launched']:
            delta_heading = (0 - ego_obs_list[5]) # ego_obs_list[5] is ego's heading
        else:
            # choose target heading according to current steps
            index = 0
            for i, interval in enumerate(self.step_list):
                if self.step <= interval:
                    index = i
                    break
            target_heading = self.target_heading_list[index]
            delta_heading = (target_heading - ego_obs_list[5])
            self.step += 1

        observation = np.zeros(8)
        observation[0] = 0                                  #  0. ego delta altitude  (unit: 1km)
        observation[1] = in_range_rad(delta_heading)        #  1. ego delta heading   (unit rad)
        observation[2] = ego_obs_list[3]                    #  2. ego_roll    (unit: rad)
        observation[3] = ego_obs_list[4]                    #  3. ego_pitch   (unit: rad)
        observation[4] = ego_obs_list[6] * 0.304 / 340      #  4. ego_v_north        (unit: mh)
        observation[5] = ego_obs_list[7] * 0.304 / 340      #  5. ego_v_east        (unit: mh)
        observation[6] = ego_obs_list[8] * 0.304 / 340      #  6. ego_v_down        (unit: mh)
        observation[7] = ego_obs_list[9] * 0.304 / 340      #  7. ego_vc        (unit: mh)
        observation = np.expand_dims(observation, axis=0)    # dim: (1,8)
        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action

    def restore(self):
        self.actor = torch.load(str(self.model_path))

    def prep_rollout(self):
        self.actor.eval()


def test_env():
    env = SingleCombatEnv(config_name='test_scwm_circle')
    agent = CircleAgent(model_path='envs\JSBSim\model\singlecontrol_baseline.pth')
    trajectory_list = []
    obs = env.reset()
    trajectory_list.append(env.render())
    cur_step = 0
    start_time = time.time()
    while True:
        cur_step += 1
        action = agent.get_action(env, env.task)
        actions = [action, np.array([20, 18.6, 20, 0])]
        obs, reward, done, env_info = env.step(actions)
        trajectory_list.append(env.render())
        if np.array(done).all():
            print(env_info)
            break
    print(time.time() - start_time)
    np.save('trajectory_data.npy', np.asarray(trajectory_list))


def grid_search_test():
    turn_seconds = [5, 4, 3]
    radius = [50, 100, 200]
    accel = [200, 400, 600]
    fly_time = [15, 20, 25]
    for s in turn_seconds:
        for r in radius:
            for a in accel:
                for t in fly_time:
                    env = SingleCombatEnv(config_name='test_scwm_circle')
                    agent = CircleAgent(model_path='envs\JSBSim\model\singlecontrol_baseline.pth')
                    obs = env.reset()
                    agent.seconds_per_turn = s
                    env.task.missile_lists[1].simulator.args.hit_distance = r
                    env.task.missile_lists[1].simulator.args.max_missile_acc = a
                    env.task.missile_lists[1].simulator.args.missile_last_time = t
                    trajectory_list = []
                    trajectory_list.append(env.render())
                    cur_step = 0
                    start_time = time.time()
                    while True:
                        cur_step += 1
                        action = agent.get_action(env, env.task)
                        actions = [action, np.array([20, 18.6, 20, 0])]
                        obs, reward, done, env_info = env.step(actions)
                        trajectory_list.append(env.render())
                        if np.array(done).all():
                            print(env_info)
                            #reward_render = env.task.reward_functions[0].get_reward_trajectory()
                            break
                    hit = env.task.missile_lists[1].missile_info[0]['strike_enm_fighter']
                    print(time.time() - start_time)
                    np.save(f'trajectory_S/{s}_{r}_{a}_{t}_{hit}.npy', np.asarray(trajectory_list))
                    env.close()


# grid_search_test()
test_env()