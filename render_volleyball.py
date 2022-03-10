from envs import slimevolley
import gym
import numpy as np
import torch
from envs.JSBSim.model.baseline_actor import *

class ACTLayer(nn.Module):
    def __init__(self, input_dim, action_dims, use_mlp_actlayer=False):
        super(ACTLayer, self).__init__()
        self._mlp_actlayer = use_mlp_actlayer
        if self._mlp_actlayer:
            self.mlp = MLPLayer(128, '128 128')
        self.action_out = Categorical(input_dim, action_dims)

    def forward(self, x):
        if self._mlp_actlayer:
            x = self.mlp(x)
        action = self.action_out(x)
        return action

class Actor(nn.Module):
    def __init__(self, input_dim=12, use_mlp_actlayer=False) -> None:
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.base = MLPBase(input_dim, '128 128')
        self.rnn = GRULayer(128, 128, 1)
        self.act = ACTLayer(128, 6, use_mlp_actlayer)
        self.to(torch.device('cpu'))

    def check(self, input):
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output

    def forward(self, obs, rnn_states):
        x = check(obs).to(**self.tpdv)
        h_s = check(rnn_states).to(**self.tpdv)
        x = self.base(x)
        x, h_s = self.rnn(x, h_s)
        actions = self.act(x)
        return actions, h_s

def _t2n(x):
    return x.detach().cpu().numpy()

num_agents = 2
render = False
ego_policy_index = 9
enm_policy_index = 0
episode_rewards = 0
ego_run_dir = "/home/lqh/jyh/CloseAirCombat/scripts/results/gym/Volleyball-v0/ppo/pfsp/run10"
enm_run_dir = "/home/lqh/jyh/CloseAirCombat/scripts/results/gym/Volleyball-v0/ppo/pfsp/run10"
ego_policy = Actor(input_dim=12, use_mlp_actlayer=True)
enm_policy = Actor(input_dim=12, use_mlp_actlayer=True)
ego_policy.eval()
enm_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))
enm_policy.load_state_dict(torch.load(enm_run_dir + f"/actor_{enm_policy_index}.pt"))

env = gym.make("Volleyball-v0")
env.seed(0)

print("Start render")
obs = env.reset()
if render:
    env.render()
ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
enm_obs =  obs[num_agents // 2:, ...]
ego_obs =  obs[:num_agents // 2, ...]
enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
while True:
    ego_actions, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states)
    ego_actions = _t2n(ego_actions)
    ego_rnn_states = _t2n(ego_rnn_states)
    enm_actions, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states)
    enm_actions = _t2n(enm_actions)
    enm_rnn_states = _t2n(enm_rnn_states)
    actions = np.concatenate((ego_actions, enm_actions), axis=1)
    # Obser reward and next obs
    obs, rewards, dones, infos = env.step(actions)
    rewards = rewards[:num_agents // 2, ...]
    episode_rewards += rewards
    if render:
        env.render()
    if dones.all():
        print(infos)
        break
    enm_obs =  obs[num_agents // 2:, ...]
    ego_obs =  obs[:num_agents // 2, ...]

print(episode_rewards)