from envs.JSBSim.envs.selfplay_env import SelfPlayEnv
import numpy as np
import pdb
import time
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog

env = SelfPlayEnv()
# aileron  elevator  rudder  throttle
env.reset()
cur_step = -1
reward_blue, reward_red = 0., 0.
start_time = time.time()
while True:
    cur_step += 1
    actions = {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': np.array([20., 18.6, 20., 0.])}
    next_obs, reward, done, env_info = env.step(actions)
    # pdb.set_trace()
    # for _ in range(100):
    #     env.sims['blue_fighter'].get_property_values(env.task.state_var)
    reward_blue += reward['blue_fighter']
    reward_red += reward['red_fighter']
    print(reward_blue, reward_red)
    if done:
        print(env_info)
        break
print(time.time() - start_time)


# def make_train_env(num_env):
#     return SubprocVecEnv([SelfPlayEnv for _ in range(num_env)])

# if __name__ == '__main__':
#     num_env = 4
#     envs = make_train_env(num_env)
#     envs.reset()
#     n_rollout = 0
#     while n_rollout < 20:
#         n_rollout += 1
#         while True:
#             actions = {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': np.array([20., 18.6, 20., 0.])}
#             next_obs, reward, done, env_info = envs.step([actions for _ in range(num_env)])
#             if np.all(done):
#                 print(env_info)
#                 break
#     envs.close()
