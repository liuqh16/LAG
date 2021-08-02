import gym
import torch


class Config(object):
    def __init__(self, env):
        # make env instance
        self.env = env  # type: gym.Env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Population-based Training
        self.thresholds = 16.
        self.top_k = 4
        self.initial_elo = 1000.
        self.ppo_hyper = [1., 1.]
        self.perturb_prob_hyper = 0.2
        self.mutate_prob_hyper = 0.8

        self.num_training_per_wider_epoch = 1
        self.device = torch.device('cpu')

        self.act_dims = 4
        self.policy_ego = [128, 128]
        self.policy_gru_config = {'num_layers': 1, 'hidden_size': 128}
        self.policy_act_mlp = [128, 128]
        self.flag_eval = False
        self.value_ego = [128, 128]
        self.value_gru_config = {'num_layers': 1, 'hidden_size': 128}
        self.value_mlp = [128, 128]

        self.lr = 3e-4
        self.discount_gamma = 0.99
        self.ppo_epoch = 4
        self.policy_clip = 0.2
        self.max_grad_norm = 2.
        self.entropy_weight = 1e-3
        self.tx_c = 3.
        self.buffer_config = {'buffer_size': 3000, 'seq_len': 8, 'batch_size': 512}
