from envs.JSBSim.envs.self_play_env import JSBSimSelfPlayEnv
import numpy as np
import pdb
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv

# env = JSBSimSelfPlayEnv()
# # aileron  elevator  rudder  throttle
# obs_flattener = DictFlattener(env.observation_space)
# env.reset()
# cur_step = -1
# reward_blue, reward_red = 0., 0.
# while True:
#     cur_step += 1
#     actions = {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': np.array([20., 18.6, 20., 0.])}
#     next_obs, reward, done, env_info = env.step(actions)
#     pdb.set_trace()
#     reward_blue += reward['blue_reward']
#     reward_red += reward['red_reward']
#     print(reward_blue, reward_red)
#     if done:
#         print(env_info)
#         break


def make_train_env():
    return SubprocVecEnv([JSBSimSelfPlayEnv for _ in range(3)])

if __name__ == '__main__':
    envs = make_train_env()
    envs.reset()
    n_rollout = 0
    while n_rollout < 20:
        n_rollout += 1
        while True:
            actions = {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': np.array([20., 18.6, 20., 0.])}
            next_obs, reward, done, env_info = envs.step([actions for _ in range(3)])
            if np.all(done):
                print(env_info)
                break
    envs.close()
