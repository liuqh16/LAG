from envs.JSBSim.envs.self_play_env import JSBSimSelfPlayEnv
import numpy as np


env = JSBSimSelfPlayEnv()
# aileron  elevator  rudder  throttle
observation = env.reset()
cur_step = -1
reward_blue, reward_red = 0., 0.
while True:
    cur_step += 1
    actions = {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': np.array([20., 18.6, 20., 0.])}
    next_obs, reward, done, env_info = env.step(actions)
    reward_blue += reward['blue_reward']
    reward_red += reward['red_reward']
    print(reward_blue, reward_red)
    if done:
        break