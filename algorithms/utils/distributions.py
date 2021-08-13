import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    @property
    def deterministic(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def log_prob(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def entropy(self):
        return super().entropy().unsqueeze(-1)


class FixedNormal(torch.distributions.Normal):
    @property
    def deterministic(self):
        return self.mean

    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Categorical, self).__init__()
        self.logits_net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logits = self.logits_net(x)
        return FixedCategorical(logits=logits)


class DiagGaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiagGaussian, self).__init__()
        self.mu_net = nn.Linear(input_dim, output_dim)
        log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.from_numpy(log_std))

    def forward(self, x):
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)
        return FixedNormal(mu, std)


if __name__ == "__main__":

    input_dim, output_dim = 10, 5

    # test Categorical
    dist_layer = Categorical(input_dim, output_dim)

    print("\n---------test Categorical.single_input---------\n")
    x = torch.rand(input_dim)
    dist = dist_layer(x)
    print(" logits:", dist.logits)
    action = dist.sample()
    print(" sample:", action)
    print(" log_prob:", dist.log_prob(action))
    print(" deterministic:", dist.deterministic)
    print(" entropy:", dist.entropy())

    print("\n---------test Categorical.batch_input---------\n")
    bt_x = torch.rand(4, input_dim)
    bt_dist = dist_layer(bt_x)
    print(" logits:", bt_dist.logits)
    action = bt_dist.sample()
    print(" sample:", action)
    print(" log_prob:", bt_dist.log_prob(action))
    print(" deterministic:", bt_dist.deterministic)
    print(" entropy:", bt_dist.entropy())

    # test DiagGaussian
    dist_layer = DiagGaussian(input_dim, output_dim)

    print("\n---------test DiagGaussian.single_input---------\n")
    x = torch.rand(input_dim)
    dist = dist_layer(x)
    action = dist.sample()
    print(" sample:", action)
    print(" log_prob:", dist.log_prob(action))
    print(" deterministic:", dist.deterministic)
    print(" entropy:", dist.entropy())

    print("\n---------test DiagGaussian.batch_input---------\n")
    bt_x = torch.rand(4, input_dim)
    bt_dist = dist_layer(bt_x)
    action = bt_dist.sample()
    print(" sample:", action)
    print(" log_prob:", bt_dist.log_prob(action))
    print(" deterministic:", bt_dist.deterministic)
    print(" entropy:", bt_dist.entropy())
