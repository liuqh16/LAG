import time
import logging
from gymnasium import spaces
import numpy as np
import torch

from lag.envs.jsbsim import MultipleCombatEnv
from lag.algorithms.ppo.ppo_actor import PPOActor


logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

def convert(obs):
    ego = 9
    offset = 6
    assert len(obs[0]) == ego + offset * 3
    single_obs = np.zeros((2,ego+offset))
    single_obs[0][:ego] = obs[0][:ego]
    single_obs[0][ego:] = obs[0][ego+offset:ego+2*offset] 
    single_obs[1][:ego] = obs[1][:ego]
    single_obs[1][ego:] = obs[1][ego+2*offset:ego+3*offset]
    return single_obs 

num_agents = 4
render = True
ego_policy_index = 1040
enm_policy_index = 1040
episode_rewards = 0
ego_run_dir = "/home/lqh/jyh/CloseAirCombat/scripts/results/MultipleCombat/2v2/NoWeapon/HierarchySelfplay/mappo/artillery_check/wandb/latest-run/files"
enm_run_dir = "/home/lqh/jyh/CloseAirCombat/scripts/results/SingleCombat/1v1/NoWeapon/HierarchySelfplay/ppo/artillery_check/wandb/latest-run/files"
experiment_name = ego_run_dir.split('/')[-4]

env = MultipleCombatEnv("2v2/NoWeapon/HierarchySelfplay")
env.seed(0)
args = Args()

ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
enm_policy = PPOActor(args, spaces.Box(low=-10, high=10., shape=(15,)), env.action_space, device=torch.device("cuda"))
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))


print("Start render")
obs, _ = env.reset()
if render:
    env.filepath = f'{experiment_name}.txt.acmi'
    env.render(mode='txt')
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((num_agents // 2, 1))
enm_obs =  obs[num_agents // 2:, :]
ego_obs =  obs[:num_agents // 2, :]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
while True:
    start = time.time()
    ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
    end = time.time()
    # print(f"NN forward time: {end-start}")
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    # 2v2 obs converst to 1v1 obs
    enm_obs = convert(enm_obs)
    enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=0)
    # Obser reward and next obs
    start = time.time()
    obs, _, rewards, dones, infos = env.step(actions)
    end = time.time()
    # print(f"Env step time: {end-start}")
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render:
        env.render(mode='txt')
    if dones.all():
        print(infos)
        break
    bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
    print(f"step:{env.current_step}, bloods:{bloods}")
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

print(episode_rewards)