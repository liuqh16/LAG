import torch
import torch.nn as nn
import gym.spaces
from mlp import MLPLayer
from distributions import Categorical, DiagGaussian, Bernoulli


class ACTLayer(nn.Module):
    def __init__(self, act_space, input_dim, hidden_size, activation_id, gain):
        super(ACTLayer, self).__init__()
        self._mlp_actlayer = False
        self._continuous_action = False
        self._multidiscrete_action = False
        self._mixed_action = False

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
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(act_space)}!")
            # TODO: discrete + continous
            self._mixed_action = True

    def forward(self, x, deterministic=False):
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

        else:
            action_dists = self.action_out(x)
            actions = action_dists.mode() if deterministic else action_dists.sample()
            action_log_probs = action_dists.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, active_masks=None, mean_entropy=True):
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
                    dist_entropy.append(action_dist.entropy())
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(dim=-1, keepdim=True)
            dist_entropy = torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True)

        else:
            action_dist = self.action_out(x)
            action_log_probs = action_dist.log_probs(action)
            if active_masks is not None:
                dist_entropy = (action_dist.entropy() * active_masks) / active_masks.sum()
            else:
                dist_entropy = action_dist.entropy()
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
        elif self._continuous_action:
            raise ValueError("Normal distribution has no `probs` attribute!")
        else:
            action_dists = self.action_out(x)
            action_probs = action_dists.probs
        return action_probs

    @property
    def output_size(self) -> int:
        if self._multidiscrete_action:
            return len(self.action_outs)
        else:
            return self.action_out.output_size


if __name__ == "__main__":

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    input_dim, batch_size = 5, 4
    act_spaces = [
        gym.spaces.Discrete(2),
        gym.spaces.Box(low=-1, high=1, shape=(2,)),
        gym.spaces.MultiDiscrete([3, 2]),
        gym.spaces.MultiBinary(2),
    ]

    for act_space in act_spaces:

        print(f"\n---------test {act_space.__class__.__name__} action space---------\n")
        actlayer = ACTLayer(act_space, input_dim, '', 1, gain=0.01)

        print("ONE")
        x = torch.rand(input_dim)
        active_masks = torch.ones(1)
        print(" probs:", actlayer.get_probs(x) if not isinstance(act_space, gym.spaces.Box) else None)
        action, log_probs = actlayer(x)
        print(" sample:", action, log_probs)
        print(" deterministic:", *actlayer(x, deterministic=True))
        assert tuple(action.size()) == (actlayer.output_size,) \
            and tuple(log_probs.size()) == (1,)

        print("BATCH")
        bt_x = torch.rand(batch_size, input_dim)
        active_masks = torch.ones(batch_size, 1)
        print(" probs:", actlayer.get_probs(bt_x) if not isinstance(act_space, gym.spaces.Box) else None)
        pre_actions = torch.as_tensor([act_space.sample() for _ in range(batch_size)], dtype=torch.float32)
        print(" pre_actions:", pre_actions)
        log_probs, entropy = actlayer.evaluate_actions(bt_x, pre_actions, active_masks)
        print(" evaluate:", torch.exp(log_probs), "\n entropy:", entropy)
        assert tuple(log_probs.size()) == (batch_size, 1) \
            and tuple(entropy.size()) == (batch_size, 1)
