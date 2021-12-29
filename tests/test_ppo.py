import sys
import os
import torch
import pytest
import numpy as np
from itertools import product
import gym.spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from config import get_config
from algorithms.ppo.ppo_actor import PPOActor
from algorithms.ppo.ppo_critic import PPOCritic
from algorithms.utils.buffer import ReplayBuffer
from algorithms.ppo.ppo_policy import PPOPolicy
from algorithms.ppo.ppo_trainer import PPOTrainer


class TestPPO:

    @pytest.mark.parametrize("obs_space, act_space, batch_size", list(product(
        [       # obs_space
            gym.spaces.Box(low=-1, high=1, shape=(18,))
        ], [    # act_space
            gym.spaces.Discrete(5),
            gym.spaces.MultiDiscrete([41, 41, 41, 30]),
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(low=-1, high=1, shape=(4,)),
        ], [    # batch_size
            1, 5
        ])))
    def test_ppo_actor(self, obs_space, act_space, batch_size):
        actor = PPOActor(get_config().parse_args(args=''), obs_space, act_space, device=torch.device("cpu"))

        obs = np.array([obs_space.sample() for _ in range(batch_size)])
        masks = np.ones((batch_size, 1))
        init_rnn_states = np.zeros((batch_size, actor.recurrent_hidden_layers, actor.recurrent_hidden_size))

        actions, action_log_probs, rnn_states = actor(obs, init_rnn_states, masks)
        assert actions.shape[0] == batch_size
        assert action_log_probs.shape == (batch_size, 1)
        assert rnn_states.shape == init_rnn_states.shape

        pre_actions = np.array([act_space.sample() for _ in range(batch_size)])
        if pre_actions.ndim < 2:
            pre_actions = np.expand_dims(pre_actions, axis=1)
        action_log_probs, dist_entropy = actor.evaluate_actions(obs, init_rnn_states, pre_actions, masks)
        assert action_log_probs.shape == (batch_size, 1)
        assert dist_entropy.shape == (batch_size, 1)

    @pytest.mark.parametrize("obs_space, batch_size", list(product(
        [       # obs_space
            gym.spaces.Box(low=-1, high=1, shape=(18,))
        ], [    # batch_size
            1, 5
        ])))
    def test_ppo_critic(self, obs_space, batch_size):
        critic = PPOCritic(get_config().parse_args(args=''), obs_space, device=torch.device("cpu"))

        obs = np.array([obs_space.sample() for _ in range(batch_size)])
        masks = np.ones((batch_size, 1))
        init_rnn_states = np.zeros((batch_size, critic.recurrent_hidden_layers, critic.recurrent_hidden_size))

        value, rnn_states = critic(obs, init_rnn_states, masks)
        assert value.shape[0] == batch_size
        assert rnn_states.shape == init_rnn_states.shape

    @pytest.mark.parametrize("num_agents, obs_space, act_space, num_mini_batch, data_chunk_length", list(product(
        [       # num_agents
            1, 2
        ], [    # obs_space
            gym.spaces.Box(low=-1, high=1, shape=(18,))
        ], [    # act_space
            gym.spaces.Discrete(5),
            gym.spaces.MultiDiscrete([41, 41, 41, 30]),
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(low=-1, high=1, shape=(4,)),
        ], [    # num_mini_batch
            1, 2
        ], [    # data_chunk_length
            1, 2,
        ])))
    def test_ppo_buffer(self, num_agents, obs_space, act_space, num_mini_batch, data_chunk_length):
        buffer = ReplayBuffer(get_config().parse_args(args=''), num_agents, obs_space, act_space)

        obs = np.array([[obs_space.sample() for _ in range(num_agents)] for _ in range(buffer.n_rollout_threads)])
        actions = np.array([[act_space.sample() for _ in range(num_agents)] for _ in range(buffer.n_rollout_threads)])
        if np.array(actions).ndim < 3:
            actions = np.expand_dims(actions, axis=-1)
        rewards = np.random.randn(buffer.n_rollout_threads, num_agents, 1)
        masks = np.ones((buffer.n_rollout_threads, num_agents, 1))
        bad_masks = np.ones((buffer.n_rollout_threads, num_agents, 1))
        action_log_probs = np.random.randn(buffer.n_rollout_threads, num_agents, 1)
        value_preds = np.random.randn(buffer.n_rollout_threads, num_agents, 1)
        rnn_states_actor = np.zeros((buffer.n_rollout_threads, num_agents, buffer.recurrent_hidden_layers, buffer.recurrent_hidden_size))
        rnn_states_critic = np.zeros((buffer.n_rollout_threads, num_agents, buffer.recurrent_hidden_layers, buffer.recurrent_hidden_size))

        buffer.insert(obs, actions, rewards, masks, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic, bad_masks)

        next_value = np.random.randn(buffer.n_rollout_threads, num_agents, 1)
        buffer.compute_returns(next_value)

        batch_count = 0
        for data in buffer.recurrent_generator(buffer, num_mini_batch, data_chunk_length):
            obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = data
            batch_count += 1
        assert batch_count == num_mini_batch

        buffer.after_update()

    @pytest.mark.parametrize("num_agents, obs_space, act_space", list(product(
        [       # num_agents
            1, 2
        ], [    # obs_space
            gym.spaces.Box(low=-1, high=1, shape=(18,))
        ], [    # act_space
            gym.spaces.Discrete(5),
            gym.spaces.MultiDiscrete([41, 41, 41, 30]),
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(low=-1, high=1, shape=(4,)),
        ])))
    def test_ppo_trainer(self, num_agents, obs_space, act_space):
        args = get_config().parse_args(args='')
        buffer = ReplayBuffer(args, num_agents, obs_space, act_space)
        policy = PPOPolicy(args, obs_space, act_space, device=torch.device("cpu"))
        trainer = PPOTrainer(args, device=torch.device("cpu"))
        policy.prep_training()
        trainer.train(policy, buffer)
