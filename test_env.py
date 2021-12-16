import pdb
import numpy as np
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv


env = SingleCombatEnv("1v1/Missile/HierarchyVsBaseline")
act_space = env.action_space[0]

obs_buf = []
act_buf = []
rew_buf = []
done_buf = []
actions_buf = []

env.seed(0)
act_space.seed(0)
obs = env.reset()
env.render()
obs_buf.append(obs.copy())

while True:
    action = [[act_space.sample()] for _ in range(env.num_agents)]
    obs, reward, done, info = env.step(action)
    env.render()
    obs_buf.append(obs.copy())
    act_buf.append(action.copy())
    rew_buf.append(reward.copy())
    done_buf.append(done.copy())
    actions_buf.append(info['actions'])

    if np.all(done):
        print(np.sum(rew_buf))
        break

t = 0
env.seed(0)
act_space.seed(0)
obs = env.reset()
assert np.linalg.norm(obs - obs_buf[t]) < 1e-8

while True:
    action = act_buf[t]
    obs, reward, done, info = env.step(action)

    try:
        assert np.linalg.norm(obs - obs_buf[t + 1]) < 1e-8 \
            and np.linalg.norm(info['actions'] - actions_buf[t]) < 1e-8 \
            and np.linalg.norm(reward - rew_buf[t]) < 1e-8 \
            and np.linalg.norm(done - done_buf[t]) < 1e-8
    except:
        pdb.set_trace()

    if np.all(done):
        break
    t += 1
