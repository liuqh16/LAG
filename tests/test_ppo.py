import sys
import os
import torch
import pytest
import numpy as np
from itertools import product
import gym.spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def create_actor(obs_space, act_space):
    from algorithms.ppo.ppo_actor import PPOActor
    from config import get_config
    return PPOActor(
        get_config().parse_args(args=''), obs_space, act_space, device=torch.device("cpu")
    )


def create_critic(obs_space):
    from algorithms.ppo.ppo_critic import PPOCritic
    from config import get_config
    return PPOCritic(get_config().parse_args(args=''), obs_space, device=torch.device("cpu"))


def create_buffer(obs_space, act_space):
    from algorithms.utils.buffer import ReplayBuffer
    from config import get_config
    parser = get_config()
    parser.add_argument('--num-agents', default=1, type=int)
    return ReplayBuffer(parser.parse_args(args=''), obs_space, act_space)


def create_trainer(obs_space, act_space):
    from algorithms.ppo.ppo_trainer import PPOTrainer
    from algorithms.ppo.ppo_policy import PPOPolicy
    from config import get_config
    parser = get_config()
    parser.add_argument('--num-agents', default=1, type=int)
    args = parser.parse_args(args='')
    policy = PPOPolicy(args, obs_space, act_space, device=torch.device("cpu"))
    return PPOTrainer(args, policy, device=torch.device("cpu"))


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
        actor = create_actor(obs_space, act_space)

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
        critic = create_critic(obs_space)

        obs = np.array([obs_space.sample() for _ in range(batch_size)])
        masks = np.ones((batch_size, 1))
        init_rnn_states = np.zeros((batch_size, critic.recurrent_hidden_layers, critic.recurrent_hidden_size))

        value, rnn_states = critic(obs, init_rnn_states, masks)
        assert value.shape[0] == batch_size
        assert rnn_states.shape == init_rnn_states.shape

    @pytest.mark.parametrize("obs_space, act_space, num_mini_batch, data_chunk_length", list(product(
        [       # obs_space
            gym.spaces.Box(low=-1, high=1, shape=(18,))
        ], [    # act_space
            gym.spaces.Discrete(5),
            gym.spaces.MultiDiscrete([41, 41, 41, 30]),
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(low=-1, high=1, shape=(4,)),
        ], [    # num_mini_batch
            1, 4
        ], [    # data_chunk_length
            1, 4, 16
        ])))
    def test_ppo_buffer(self, obs_space, act_space, num_mini_batch, data_chunk_length):
        buffer = create_buffer(obs_space, act_space)

        obs = [obs_space.sample() for _ in range(buffer.n_rollout_threads)]
        actions = [act_space.sample() for _ in range(buffer.n_rollout_threads)]
        if np.array(actions).ndim < 2:
            actions = np.expand_dims(actions, axis=1)
        rewards = np.random.randn(buffer.n_rollout_threads, 1)
        masks = np.ones((buffer.n_rollout_threads, 1))
        bad_masks = np.ones((buffer.n_rollout_threads, 1))
        action_log_probs = np.random.randn(buffer.n_rollout_threads, 1)
        value_preds = np.random.randn(buffer.n_rollout_threads, 1)
        rnn_states_actor = np.zeros((buffer.n_rollout_threads, buffer.recurrent_hidden_layers, buffer.recurrent_hidden_size))
        rnn_states_critic = np.zeros((buffer.n_rollout_threads, buffer.recurrent_hidden_layers, buffer.recurrent_hidden_size))

        buffer.insert(obs, actions, rewards, masks, bad_masks, action_log_probs, value_preds, rnn_states_actor, rnn_states_critic)

        next_value = np.random.randn(buffer.n_rollout_threads, 1)
        buffer.compute_returns(next_value)

        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        batch_count = 0
        for data in buffer.recurrent_generator(advantages, num_mini_batch, data_chunk_length):
            obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
                returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = data
            batch_count += 1
        assert batch_count == num_mini_batch

        buffer.after_update()

    @pytest.mark.parametrize("obs_space, act_space", list(product(
        [       # obs_space
            gym.spaces.Box(low=-1, high=1, shape=(18,))
        ], [    # act_space
            gym.spaces.Discrete(5),
            gym.spaces.MultiDiscrete([41, 41, 41, 30]),
            gym.spaces.MultiBinary(4),
            gym.spaces.Box(low=-1, high=1, shape=(4,)),
        ])))
    def test_ppo_trainer(self, obs_space, act_space):
        buffer = create_buffer(obs_space, act_space)
        trainer = create_trainer(obs_space, act_space)
        trainer.prep_training()
        trainer.train(buffer)
