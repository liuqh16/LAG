import torch
import torch.nn as nn

from utils import init

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

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

    @property
    def output_size(self) -> int:
        return 1


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(DiagGaussian, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(torch.zeros(num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        action_mean = self.fc_mean(x)
        return FixedNormal(action_mean, self.logstd.exp())

    @property
    def output_size(self) -> int:
        return self._num_outputs


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, gain=0.01):
        super(Bernoulli, self).__init__()

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self._num_outputs = num_outputs

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

    @property
    def output_size(self) -> int:
        return self._num_outputs


if __name__ == "__main__":

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    num_inputs, num_outputs, batch_size = 10, 5, 4
    distributions = [Categorical, DiagGaussian, Bernoulli]
    # distributions = [DiagGaussian]

    for dist_builder in distributions:

        dist_layer = dist_builder(num_inputs, num_outputs)

        print(f"\n---------test {dist_builder.__name__}.single_input---------\n")
        x = torch.rand(num_inputs)
        dist = dist_layer(x)
        action = dist.sample()
        print(f" sample{tuple(action.size())}:\n ", action)
        log_probs = dist.log_probs(action)
        print(f" log_probs{tuple(log_probs.size())}:\n ", log_probs)
        mode = dist.mode()
        print(f" mode{tuple(mode.size())}:\n ", mode)
        entropy = dist.entropy()
        print(f" entropy{tuple(entropy.size())}:\n ", entropy)

    for dist_builder in distributions:

        dist_layer = dist_builder(num_inputs, num_outputs)

        print(f"\n---------test {dist_builder.__name__}.batch_input---------\n")
        bt_x = torch.rand(batch_size, num_inputs)
        bt_dist = dist_layer(bt_x)
        action = bt_dist.sample()
        print(f" sample{tuple(action.size())}:\n ", action)
        log_probs = bt_dist.log_probs(action)
        print(f" log_probs{tuple(log_probs.size())}:\n ", log_probs)
        mode = bt_dist.mode()
        print(f" mode{tuple(mode.size())}:\n ", mode)
        entropy = bt_dist.entropy()
        print(f" entropy{tuple(entropy.size())}:\n ", entropy)
