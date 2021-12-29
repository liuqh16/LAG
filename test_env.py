import numpy as np
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
import logging

parallel_num = 1
envs = DummyVecEnv([lambda: SingleCombatEnv("1v1/DodgeMissile/vsBaseline") for _ in range(parallel_num)])

# DataType test
obs_shape = (parallel_num, envs.num_agents, *envs.observation_space.shape)
act_shape = (parallel_num, envs.num_agents, *envs.action_space.shape)
reward_shape = (parallel_num, envs.num_agents, 1)
done_shape = (parallel_num, envs.num_agents, 1)

obss = envs.reset()
assert obss.shape == obs_shape

actions = np.array([[envs.action_space.sample() for _ in range(envs.num_agents)] for _ in range(parallel_num)])
while True:
    obss, rewards, dones, infos = envs.step(actions)
    assert obss.shape == obs_shape and actions.shape == act_shape \
        and rewards.shape == reward_shape and dones.shape == done_shape \
        and infos.shape[0] == parallel_num and isinstance(infos[0], dict)
    # terminate if any of the parallel envs has been done
    if np.any(dones):
        break
envs.close()
