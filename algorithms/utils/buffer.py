import numpy as np
import torch
from .util import get_shape_from_space

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 0, *range(2, x.ndim)).reshape(-1, *x.shape[2:])


class ReplayBuffer(object):
    def __init__(self, args, num_agents, obs_space, act_space):
        # env config
        self.n_rollout_threads = args.n_rollout_threads
        # buffer config
        self.gamma = args.gamma
        self.episode_length = args.episode_length
        self.use_gae = args.use_gae
        self.gae_lambda = args.gae_lambda
        # rnn config
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.data_chunk_length = args.data_chunk_length

        obs_shape = get_shape_from_space(obs_space)
        act_shape = get_shape_from_space(act_space)

        # (o_0, a_0, r_0, d_0, ..., o_T)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.dones = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        # pi(a)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, *act_shape), dtype=np.float32)
        # v(o), r(o) while advantage = returns - value_preds
        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        # rnn
        self.rnn_states_actor = np.zeros((self.episode_length + 1, self.n_rollout_threads, \
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
            value_preds:        value(o_{t})   # NOTE: not value(o_{t+1})
            rnn_states_actor:   ha_{t+1}
            rnn_states_critic:  hc_{t+1}
        """
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.rewards[self.step] = rewards.copy()
        self.dones[self.step] = dones.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()

        self.step = (self.step + 1) % self.episode_length

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
        assert self.n_rollout_threads * self.episode_length >= self.data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(self.n_rollout_threads, self.episode_length, self.data_chunk_length))

        # Transpose and reshape parallel data into sequential data
        obs = _cast(self.obs[:-1])
        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        dones = _cast(self.dones[:-1])
        rnn_states_actor = _cast(self.rnn_states_actor[:-1])
        rnn_states_critic = _cast(self.rnn_states_critic[:-1])

        buffer_size = self.episode_length * self.n_rollout_threads
        data_chunks = buffer_size // self.data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch
        flag_splits = (np.where(self.dones == True)[0] + 1).tolist()

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        for indices in sampler:

            share_obs_batch = []
            obs_batch = []

            rnn_states_actor_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            dones_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * self.data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                obs_batch.append(obs[ind:ind+self.data_chunk_length])
                actions_batch.append(actions[ind:ind+self.data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+self.data_chunk_length])
                return_batch.append(returns[ind:ind+self.data_chunk_length])
                dones_batch.append(dones[ind:ind+self.data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+self.data_chunk_length])
                adv_targ.append(advantages[ind:ind+self.data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = self.data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim) 
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            dones_batch = np.stack(dones_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])
            
            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            dones_batch = _flatten(L, N, dones_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch



