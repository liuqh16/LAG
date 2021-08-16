import pdb
import time
import numpy as np
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def test_env():
    env = SingleCombatEnv(config='singlecombat_simple')
    # env = SingleCombatEnv(config='singlecombat_with_missile')
    act_space = env.action_space[0]

    env.reset()
    cur_step = -1
    episode_reward = np.zeros(env.num_agents)
    start_time = time.time()
    while True:
        cur_step += 1
        # flying straight forward
        actions = [np.array([20, 18.6, 20, 0]) for _ in range(env.num_agents)]
        # random fly
        # actions = {"red_fighter": act_space.sample(), 'blue_fighter': act_space.sample()}
        next_obs, reward, done, env_info = env.step(actions)
        episode_reward += reward
        print(episode_reward)
        if done:
            print(env_info)
            break
    print(time.time() - start_time)


def test_parallel_env():
    
    def make_train_env(num_env, config='singlecombat_simple'):
        def env_fn():
            return SingleCombatEnv(config=config)
        return DummyVecEnv([env_fn for _ in range(num_env)])

    start_time = time.time()
    num_env = 2
    envs = make_train_env(num_env)
    act_space = envs.action_space[0]
    num_agents = len(envs.action_space)

    n_total_steps = 2000
    n_current_steps = 0
    n_current_episodes = 0
    obss = envs.reset()
    while n_current_steps < n_total_steps:
        actions = [[act_space.sample() for _ in range(num_agents)] for _ in range(num_env)]
        next_obss, rewards, dones, env_infos = envs.step(actions)
        new_samples = list(zip(obss, actions, rewards, next_obss, dones))
        n_current_steps += len(new_samples)
        for i, done in enumerate(dones):
            if done:
                n_current_episodes += 1
    print(f"Collect data finish: total step {n_current_steps}, total episode {n_current_episodes}, timecost: {time.time() - start_time:.2f}s")
    envs.close()


# test_env()
test_parallel_env()
