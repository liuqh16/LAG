import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces
from .mlp import MLPLayer
from .distributions import Categorical, DiagGaussian


class ACTLayer(nn.Module):
    def __init__(self, act_space, input_dim, hidden_size, activation_id):
        super(ACTLayer, self).__init__()
        self._continuous_action = False
        self._multidiscrete_action = False
        self._mlp_actlayer = False

        if len(hidden_size) > 0:
            self._mlp_actlayer = True
            self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
            input_dim = self.mlp.output_size

        if isinstance(act_space, gym.spaces.Discrete):
            action_dim = act_space.n
            self.action_out = Categorical(input_dim, action_dim)
        elif isinstance(act_space, gym.spaces.Box):
            self._continuous_action = True
            action_dim = act_space.shape[0]
            self.action_out = DiagGaussian(input_dim, action_dim)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self._multidiscrete_action = True
            action_dims = act_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(Categorical(input_dim, action_dim))
            self.action_outs = nn.ModuleList(action_outs)
        else:  # TODO: discrete + continous
            raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")
    
    def forward(self, x, deterministic=False):
        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action = action_dist.deterministic if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1)
        elif self._continuous_action:
            action_dists = self.action_out(x)
            actions = action_dists.deterministic if deterministic else action_dists.sample() 
            action_log_probs = action_dists.log_prob(actions)
        else:
            action_dists = self.action_out(x)
            actions = action_dists.deterministic if deterministic else action_dists.sample() 
            action_log_probs = action_dists.log_prob(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action):
        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_dist = action_out(x)
                action_log_probs.append(action_dist.log_prob(act))
                dist_entropy.append(action_dist.entropy().mean())
            action_log_probs = torch.cat(action_log_probs, dim=-1)
            dist_entropy = torch.tensor(dist_entropy).mean()
        elif self._continuous_action:
            action_dists = self.action_out(x)
            action_log_probs = action_dists.log_prob(action)
            dist_entropy = action_dists.entropy().mean()       
        else:
            action_dists = self.action_out(x)
            action_log_probs = action_dists.log_prob(action)
            dist_entropy = action_dists.entropy().mean()
        return action_log_probs, dist_entropy

    def get_probs(self, x):
        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action_prob = action_dist.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, dim=-1)
        elif self._continuous_action:
            assert ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x)
            action_probs = action_dists.probs
        return action_probs


if __name__ == "__main__":
    input_dim = 5
    print("\n---------test Discrete action space---------\n")
    act_space = gym.spaces.Discrete(3)
    actlayer = ACTLayer(act_space, input_dim, 0, 0, 0)
    # print(actlayer)
    print("ONE")
    x = torch.rand(input_dim)
    print(" probs:", actlayer.get_probs(x))
    action, log_probs = actlayer(x)
    print(" sample:", action, log_probs)
    print(" deterministic:", actlayer(x, deterministic=True))
    print("BATCH")
    cur_inputs = torch.rand(4, input_dim)
    print(" probs:", actlayer.get_probs(cur_inputs))
    pre_actions = torch.as_tensor([act_space.sample() for _ in range(4)])
    print(" pre_actions:", pre_actions)
    log_probs, entropy = actlayer.evaluate_actions(cur_inputs, pre_actions)
    print(" evaluate:\n", torch.exp(log_probs), "\n mean_entropy:", entropy.detach().numpy())

    print("\n---------test MultiDiscrete action space---------\n")
    act_space = gym.spaces.MultiDiscrete([3, 2])
    actlayer = ACTLayer(act_space, input_dim, 0, 0, 0)
    # print(actlayer)
    print("ONE")
    x = torch.rand(input_dim)
    print(" probs:", actlayer.get_probs(x))
    action, log_probs = actlayer(x)
    print(" sample:", action, log_probs)
    print(" deterministic:", actlayer(x, deterministic=True))
    print("BATCH")
    cur_inputs = torch.rand(4, input_dim)
    print(" probs:", actlayer.get_probs(cur_inputs))
    pre_actions = torch.as_tensor([act_space.sample() for _ in range(4)])
    print(" pre_actions:", pre_actions)
    log_probs, entropy = actlayer.evaluate_actions(cur_inputs, pre_actions)
    print(" evaluate:\n", torch.exp(log_probs), "\n mean_entropy:", entropy.detach().numpy())

    print("\n---------test Box action space---------\n")
    act_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
    actlayer = ACTLayer(act_space, input_dim, 0, 0, 0)
    # print(actlayer)
    print("ONE")
    x = torch.rand(input_dim)
    action, log_probs = actlayer(x)
    print(" sample:", action, log_probs)
    print(" deterministic:", actlayer(x, deterministic=True))
    print("BATCH")
    cur_inputs = torch.rand(4, input_dim)
    pre_actions = torch.as_tensor([act_space.sample() for _ in range(4)])
    print(" pre_actions:", pre_actions)
    log_probs, entropy = actlayer.evaluate_actions(cur_inputs, pre_actions)
    print(" evaluate:\n", torch.exp(log_probs), "\n mean_entropy:", entropy.detach().numpy())
