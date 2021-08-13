import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.util import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        self._hidden_size = args.hidden_size
        self._act_hidden_size = args.act_hidden_size
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_hidden_size = args.recurrent_hidden_size
        self._recurrent_hidden_layers = args.recurrent_hidden_layers
        self._activation_id = args.activation_id
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.base = MLPBase(args, obs_space)

        input_size = self.base.output_size
        if self._use_recurrent_policy:
            self.rnn = GRULayer(input_size, self._recurrent_hidden_size, self._recurrent_hidden_layers)
            input_size = self.rnn.output_size

        if len(self._act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self._act_hidden_size, self._activation_id)

        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        critic_features = self.base(obs)

        if self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states)

        if len(self._act_hidden_size) > 0:
            critic_features = self.mlp(obs)

        values = self.value_out(critic_features)

        return values, rnn_states
