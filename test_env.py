import numpy as np
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
import logging
import time
logging.basicConfig(level=logging.DEBUG)

def test_env():
    parallel_num = 1
    envs = DummyVecEnv([lambda: SingleCombatEnv("1v1/NoWeapon/HierarchySelfplay") for _ in range(parallel_num)])

    envs.reset()
    # DataType test
    obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
    # act_shape = (parallel_num, envs.num_agents, *envs.action_space.shape)
    reward_shape = (parallel_num, envs.num_agents, 1)
    done_shape = (parallel_num, envs.num_agents, 1)


    def convert(sample):
        return np.concatenate((sample[0], np.expand_dims(sample[1], axis=0)))

    episode_reward = 0
    step = 0
    while True:
        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        obss, rewards, dones, infos = envs.step(actions)
        bloods = [envs.envs[0].agents[agent_id].bloods for agent_id in envs.envs[0].agents.keys()]
        print(f"step:{step}, bloods:{bloods}")
        episode_reward += rewards[:,0,:]
        envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
        # terminate if any of the parallel envs has been done
        if np.all(dones):
            print(episode_reward)
            break
        step += 1
    envs.close()

def test_multi_env():
    parallel_num = 1
    envs = ShareDummyVecEnv([lambda: MultipleCombatEnv('2v2/NoWeapon/HierarchySelfplay') for _ in range(parallel_num)])
    assert envs.num_agents == 4
    obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
    share_obs_shape = (parallel_num, envs.num_agents, *envs.share_observation_space.shape)
    reward_shape = (parallel_num, envs.num_agents, 1)
    done_shape = (parallel_num, envs.num_agents, 1)

    # DataType test
    obs, share_obs = envs.reset()
    step = 0
    envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
    assert obs.shape == obs_shape and share_obs.shape == share_obs_shape
    while True:
        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        start = time.time()
        obs, share_obs, rewards, dones, info = envs.step(actions)
        bloods = [envs.envs[0].agents[agent_id].bloods for agent_id in envs.envs[0].agents.keys()]
        print(f"step:{step}, bloods:{bloods}")
        end = time.time()
        # print(rewards)
        envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
        assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
        if np.all(dones):
            break
        step += 1

    envs.close()

test_multi_env()
