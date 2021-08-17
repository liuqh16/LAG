import numpy as np
import torch
from .util import get_shape_from_space

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, *range(3, x.ndim)).reshape(-1, *x.shape[3:])


class ReplayBuffer(object):
    def __init__(self, args, num_agents, obs_space, act_space):
        # env config
        self.num_agents = num_agents
        self.n_rollout_threads = args.n_rollout_threads
        # buffer config
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.data_chunk_length = args.data_chunk_length

        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_0, ..., o_T)
        self.obs = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # pi(a)
        self.action_log_probs = np.zeros((self.buffer_size, self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        # v(o), r(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.buffer_size + 1, self.n_rollout_threads, self.num_agents, \
            self.recurrent_hidden_layers, self.recurrent_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states_actor)

        self.step = 0

    def insert(self, obs, actions, rewards, dones, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic):
        """Insert numpy data.
        Args:
            obs:                o_{t+1}
            actions:            a_{t}
            rewards:            r_{t}
            dones:              done_{t}
            action_log_probs:   log_prob(a_{t})
            value_preds:        value(o_{t})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """
        self.obs[self.step + 1] = obs.copy().reshape(self.obs.shape[1:])
        self.actions[self.step] = actions.copy().reshape(self.actions.shape[1:])
        self.rewards[self.step] = rewards.copy().reshape(self.rewards.shape[1:])
        self.dones[self.step] = dones.copy().reshape(self.dones.shape[1:])
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()

        self.step = (self.step + 1) % self.buffer_size

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()

    def compute_returns(self, next_value):
        if self.use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                td_delta = self.rewards[step] + self.gamma * (1 - self.dones[step]) * self.value_preds[step + 1] - self.value_preds[step]
                gae = td_delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.rewards[step] + self.gamma * (1 - self.dones[step]) * self.returns[step + 1]

    def recurrent_generator(self, advantages, num_mini_batch):
        """A recurrent generator that returns a dictionary providing training data arranged in mini batches.
        This generator shuffles the data by sequences.
        """
        assert self.n_rollout_threads * self.buffer_size >= self.data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(self.n_rollout_threads, self.buffer_size, self.data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = _cast(self.obs[:-1])
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        returns = _cast(self.returns[:-1])
        dones = _cast(self.dones)
        value_preds = _cast(self.value_preds[:-1])
        rnn_states_actor = _cast(self.rnn_states_actor[:-1])
        rnn_states_critic = _cast(self.rnn_states_critic[:-1])

        # Split episodes and get chunk indices
        done_splits = np.where(dones == True)[0] + 1
        buffer_splits = (np.arange(self.n_rollout_threads * self.num_agents + 1)) * self.buffer_size
        chunk_splits = np.unique(np.concatenate([done_splits, buffer_splits]))
        chunk_splits = np.concatenate([np.arange(chunk_splits[i], chunk_splits[i+1], self.data_chunk_length) \
            for i in range(len(chunk_splits) - 1)] + [chunk_splits[-1:]], axis=-1)
        data_chunk_indices = np.array(list(zip(chunk_splits[:-1], chunk_splits[1:])))

        # Get batch size and shuffle chunk data
        data_chunks = data_chunk_indices.shape[0]
        mini_batch_size = data_chunks // num_mini_batch
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        # Generate batch data
        T, N = self.data_chunk_length, mini_batch_size

        for indices in sampler:

            # These are all from_numpys of size (L, N, Dim) 
            obs_batch = np.zeros((T, N, *obs.shape[1:]))
            actions_batch = np.zeros((T, N, *actions.shape[1:]))
            old_action_log_probs_batch = np.zeros((T, N, *action_log_probs.shape[1:]))
            advantages_batch = np.zeros((T, N, *advantages.shape[1:]))
            return_batch = np.zeros((T, N, *returns.shape[1:]))
            dones_batch = np.zeros((T, N, *dones.shape[1:]))
            value_preds_batch = np.zeros((T, N, *value_preds.shape[1:]))

            # RNN states is just a (N, -1) from_numpy
            rnn_states_actor_batch = np.zeros((N, *rnn_states_actor.shape[1:]))
            rnn_states_critic_batch = np.zeros((N, *rnn_states_critic.shape[1:]))

            for N_idx, c in enumerate(indices):

                T_start, T_end = data_chunk_indices[c]
                obs_batch[:T_end-T_start, N_idx] = obs[T_start:T_end]
                actions_batch[:T_end-T_start, N_idx] = actions[T_start:T_end]
                value_preds_batch[:T_end-T_start, N_idx] = value_preds[T_start:T_end]
                return_batch[:T_end-T_start, N_idx] = returns[T_start:T_end]
                dones_batch[:T_end-T_start, N_idx] = dones[T_start:T_end]
                old_action_log_probs_batch[:T_end-T_start, N_idx] = action_log_probs[T_start:T_end]
                advantages_batch[:T_end-T_start, N_idx] = advantages[T_start:T_end]

                rnn_states_actor_batch[N_idx] = rnn_states_actor[T_start]
                rnn_states_critic_batch[N_idx] = rnn_states_critic[T_start]

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            dones_batch = _flatten(T, N, dones_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            advantages_batch = _flatten(T, N, advantages_batch)

            yield obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, \
                value_preds_batch, return_batch, old_action_log_probs_batch, advantages_batch
