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


if __name__ == "__main__":
    import numpy as np
    import gym.spaces
    from ...config import get_config
    parser = get_config()
    all_args = parser.parse_args()

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(18,))
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])
    actor = PPOActor(all_args, obs_space, act_space)

    print("ONE")
    obs = np.expand_dims(obs_space.sample(), axis=0)
    print(" obs shape:", obs.shape)
    init_rnn_state = np.zeros((1, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size))
    action, action_log_prob, rnn_state = actor(obs, init_rnn_state)
    print(" action:", action, "shape:", action.shape)
    print(" action_log_prob:", action_log_prob)
    print(" rnn_state shape:", rnn_state.shape)
    pre_action = np.expand_dims(act_space.sample(), axis=0)
    action_log_prob, dist_entropy = actor.evaluate_actions(obs, init_rnn_state, pre_action)
    print(" action_log_prob:", action_log_prob)
    print(" dist_entropy:", dist_entropy)

    print("BATCH")
    obss = np.array([obs_space.sample() for _ in range(5)])
    print(" obs shape:", obss.shape)
    init_rnn_states = np.zeros((5, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size))
    actions, action_log_probs, rnn_states = actor(obss, init_rnn_states)
    print(" action:", actions, "shape:", actions.shape)
    print(" action_log_probs:", action_log_probs)
    print(" rnn_state shape:", rnn_states.shape)
    pre_actions = np.array([act_space.sample() for _ in range(5)])
    action_log_probs, dist_entropy = actor.evaluate_actions(obss, init_rnn_states, pre_actions)
    print(" action_log_probs:", action_log_probs)
    print(" dist_entropy:", dist_entropy)
