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
    env.render()
    next_obs, reward, done, env_info = env.step(actions)
    if env.task.all_type_rewards['blue_fighter']['blood'] > 0:
        print(10)
    reward_blue += reward['blue_reward']
    reward_red += reward['red_reward']
    if done:
        break
print(reward_blue, reward_red)