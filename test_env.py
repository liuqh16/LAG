import numpy as np
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c


env = MultipleCombatEnv("2v2/NoWeapon/Selfplay")
assert env.num_agents == 4

# DataType test
obs, share_obs = env.reset()

while True:
    actions = [env.action_space.sample() for _ in range(env.num_agents)]
    obs, share_obs, rewards, dones, info = env.step(actions)

    # save previous data
    if all(dones.values()):
        print(info)
        break
