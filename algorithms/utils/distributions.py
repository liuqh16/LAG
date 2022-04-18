import torch
import torch.nn as nn

from .utils import init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

# Standardize distribution interfaces


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        # Single: [1] => [] => [] => [1, 1] => [1] => [1]
        # Batch: [N]/[N, 1] => [N] => [N] => [N, 1] => [N] => [N, 1]
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.squeeze(-1).unsqueeze(-1).size())
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        # Single: [K] => [K] => [1]
        # Batch: [N, K] => [N, K] => [N, 1]
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Categorical, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.logits_net(x)
        return FixedCategorical(logits=x)

    @property
    def output_size(self) -> int:
        return 1


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(DiagGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.mu_net = init_(nn.Linear(num_inputs, num_outputs))
        self.log_std = nn.Parameter(torch.zeros(num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        action_mean = self.mu_net(x)
        return FixedNormal(action_mean, self.log_std.exp())

    @property
    def output_size(self) -> int:
        return self._num_outputs

class BetaShootBernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(BetaShootBernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs
        self.constraint = nn.Softplus()

    def forward(self, x, **kwargs):
        x = self.net(x)
        x = self.constraint(x) # contrain alpha, beta >=0
        x = 100 - self.constraint(100-x) # constrain alpha, beta <=100
        alpha = 1 + x[:, 0].unsqueeze(-1)
        beta = 1 + x[:, 1].unsqueeze(-1)
        alpha_0 = kwargs['alpha0']
        beta_0 = kwargs['beta0']
        # print(f"{alpha}, {beta}, {alpha_0}, {beta_0}")
        p = (alpha + alpha_0) / (alpha + alpha_0 + beta + beta_0)
        return FixedBernoulli(p)

    @property
    def output_size(self) -> int:
        return self._num_outputs

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Bernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.logits_net = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        x = self.logits_net(x)
        return FixedBernoulli(logits=x)

    @property
    def output_size(self) -> int:
        return self._num_outputs
