import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.util import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
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

        self.act = ACTLayer(act_space, input_size, self._act_hidden_size, self._activation_id)

        self.to(device)

    def forward(self, obs, rnn_states, deterministic=False):        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states)

        actions, action_log_probs = self.act(actor_features, deterministic)
        
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action):
        obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action)

        return action_log_probs, dist_entropy
