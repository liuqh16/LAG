import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.util import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        critic_features = self.base(obs)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states


if __name__ == "__main__":
    import numpy as np
    import gym.spaces
    from ...config import get_config
    parser = get_config()
    all_args = parser.parse_args()

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(18,))
    critic = PPOCritic(all_args, obs_space)

    print("ONE")
    obs = np.expand_dims(obs_space.sample(), axis=0)
    print(" obs shape:", obs.shape)
    init_rnn_state = np.zeros((1, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size))
    value, rnn_state = critic(obs, init_rnn_state)
    print(" value:", value, "shape:", value.shape)
    print(" rnn_state shape:", rnn_state.shape)

    print("BATCH")
    obss = np.array([obs_space.sample() for _ in range(5)])
    print(" obs shape:", obss.shape)
    init_rnn_states = np.zeros((5, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size))
    values, rnn_states = critic(obss, init_rnn_states)
    print(" values:", values, "shape:", values.shape)
    print(" rnn_state shape:", rnn_states.shape)
