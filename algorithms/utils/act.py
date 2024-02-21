import gymnasium as gym
import torch
import torch.nn as nn

from .distributions import BetaShootBernoulli, Categorical, DiagGaussian, Bernoulli
from .mlp import MLPLayer


class ACTLayer(nn.Module):
    def __init__(self, act_space, input_dim, hidden_size, activation_id, gain):
        super(ACTLayer, self).__init__()
        self._mlp_actlayer = False
        self._continuous_action = False
        self._multidiscrete_action = False
        self._mixed_action = False
        self._shoot_action = False

        if len(hidden_size) > 0:
            self._mlp_actlayer = True
            self.mlp = MLPLayer(input_dim, hidden_size, activation_id)
            input_dim = self.mlp.output_size

        if isinstance(act_space, gym.spaces.Discrete):
            action_dim = act_space.n
            self.action_out = Categorical(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.Box):
            self._continuous_action = True
            action_dim = act_space.shape[0]
            self.action_out = DiagGaussian(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.MultiBinary):
            action_dim = act_space.shape[0]
            self.action_out = Bernoulli(input_dim, action_dim, gain)
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            self._multidiscrete_action = True
            action_dims = act_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(Categorical(input_dim, action_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
        elif isinstance(act_space, gym.spaces.Tuple) and  \
              isinstance(act_space[0], gym.spaces.MultiDiscrete) and \
                  isinstance(act_space[1], gym.spaces.Discrete):
            # NOTE: only for shoot missile
            self._shoot_action = True
            discrete_dims = act_space[0].nvec
            self._discrete_dim = act_space[0].shape[0]
            self._control_shoot_dim = 2
            self._shoot_dim = 1
            action_outs = []
            for discrete_dim in discrete_dims:
                action_outs.append(Categorical(input_dim, discrete_dim, gain))
            action_outs.append(BetaShootBernoulli(input_dim, self._control_shoot_dim, gain))
            self.action_outs = nn.ModuleList(action_outs)
        else: 
            raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")

    def forward(self, x, deterministic=False, **kwargs):
        """
        Compute actions and action logprobs from given input.

        Args:
            x (torch.Tensor): input to network.
            deterministic (bool): whether to sample from action distribution or return the mode.

        Returns:
            actions (torch.Tensor): actions to take.
            action_log_probs (torch.Tensor): log probabilities of taken actions.
        """
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
        
        elif self._shoot_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs[:-1]:
                action_dist = action_out(x)
                action = action_dist.mode() if deterministic else action_dist.sample()
                action_log_prob = action_dist.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            shoot_action_dist = self.action_outs[-1](x, **kwargs)
            shoot_action = shoot_action_dist.mode() if deterministic else shoot_action_dist.sample()
            actions.append(shoot_action)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dists = self.action_out(x)
            actions = action_dists.mode() if deterministic else action_dists.sample()
            action_log_probs = action_dists.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, active_masks=None, **kwargs):
        """
        Compute log probability and entropy of given actions.

        Args:
            x (torch.Tensor): input to network.
            action (torch.Tensor): actions whose entropy and log probability to evaluate.
            active_masks (torch.Tensor): denotes whether an agent is active or dead.

        Returns:
            action_log_probs (torch.Tensor): log probabilities of the input actions.
            dist_entropy (torch.Tensor): action distribution entropy for the given inputs.
        """
        if self._mlp_actlayer:
            x = self.mlp(x)

        if self._multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_dist = action_out(x)
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        elif self._shoot_action:
            dis_action, shoot_action = action.split((self._discrete_dim, self._shoot_dim), dim=-1)
            action_log_probs = []
            dist_entropy = []
            # multi-discrete action
            dis_action = torch.transpose(dis_action, 0, 1)
            for action_out, act in zip(self.action_outs[:-1], dis_action):
                action_dist = action_out(x)
                action_log_probs.append(action_dist.log_probs(act.unsqueeze(-1)))
                if active_masks is not None:
                    dist_entropy.append((action_dist.entropy() * active_masks) / active_masks.sum())
                else:
                    dist_entropy.append(action_dist.entropy() / action_log_probs[-1].size(0))

            # shoot action
            shoot_action_dist = self.action_outs[-1](x, **kwargs)
            action_log_probs.append(shoot_action_dist.log_probs(shoot_action))
            if active_masks is not None:
                dist_entropy.append((shoot_action_dist.entropy() * active_masks) / active_masks.sum())
            else:
                dist_entropy.append(shoot_action_dist.entropy() / action_log_probs[-1].size(0))

            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks) / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy() / action_log_probs.size(0)
        return action_log_probs, dist_entropy

    def get_probs(self, x):
        """
        Compute action probabilities from inputs.

        Args:
            x (torch.Tensor): input to network.

        Return:
            action_probs (torch.Tensor):
        """
        if self._mlp_actlayer:
            x = self.mlp(x)
        if self._multidiscrete_action:
            action_probs = []
            for action_out in self.action_outs:
                action_dist = action_out(x)
                action_prob = action_dist.probs
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs, dim=-1)
        elif self._continuous_action or self._shoot_action:
            raise ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x)
            action_probs = action_dists.probs
        return action_probs

    @property
    def output_size(self) -> int:
        if self._multidiscrete_action or self._shoot_action:
            return len(self.action_outs)
        else:
            return self.action_out.output_size
