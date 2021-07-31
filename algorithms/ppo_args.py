import torch
import gym
import numpy as np
# import envs
# from envs.collections_task.self_play_task import SelfPlayTask
# from envs.collections_env.self_play_env import JSBSimEnvSelfEnv


class Config(object):
    def __init__(self):
        # Parallel Training
        self.num_agents = 5
        self.num_parallel_each_agent = 7
        self.eval_num = 10

        self.thresholds = 16.
        self.top_k = 4
        self.initial_elo = 1000.
        # Population-based Training
        self.reward_hyper = [1.]
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
        self.buffer_config = {'buffer_size': 7200, 'seq_len': 8, 'batch_size': 512}










