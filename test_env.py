import numpy as np
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv
from envs.JSBSim.core.catalog import Catalog as c
import logging
import time
logging.basicConfig(level=logging.DEBUG)

def test_multi_env():
    parallel_num = 4
    envs = ShareSubprocVecEnv([lambda: MultipleCombatEnv('2v2/NoWeapon/HierarchySelfplay') for _ in range(parallel_num)])
    assert envs.num_agents == 4
    obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
    share_obs_shape = (parallel_num, envs.num_agents, *envs.share_observation_space.shape)
    reward_shape = (parallel_num, envs.num_agents, 1)
    done_shape = (parallel_num, envs.num_agents, 1)

    # DataType test
    obs, share_obs = envs.reset()
    # envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
    assert obs.shape == obs_shape and share_obs.shape == share_obs_shape
    while True:
        actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
        start = time.time()
        obs, share_obs, rewards, dones, info = envs.step(actions)
        end = time.time()
        # print(rewards)
        # envs.render(mode='txt', filepath='JSBSimRecording.txt.acmi')
        assert obs.shape == obs_shape and rewards.shape == reward_shape and dones.shape == done_shape and share_obs_shape
        if np.all(dones):
            break

    envs.close()


test_multi_env()