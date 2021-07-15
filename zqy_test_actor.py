from envs.JSBSim.envs.selfplay_env import SelfPlayEnv
from algorithms.ppo_actor import PolicyRnnMultiHead
import numpy as np
import pdb
import time
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.flatten_utils import DictFlattener
import torch


def make_train_env(num_env):
    return SubprocVecEnv([SelfPlayEnv for _ in range(num_env)])


if __name__ == '__main__':
    env = SelfPlayEnv()
    num_env = 5
    act_space = env.action_space


    class Config(object):
        def __init__(self):
            self.act_dims = 10
            self.policy_ego = [32, 32]
            self.policy_gru_config = {'num_layers': 1, 'hidden_size': 64}
            self.policy_act_mlp = [32, 32]
            self.flag_eval = False
            self.device = torch.device('cpu')

    actor = PolicyRnnMultiHead(Config(), env.observation_space, env.action_space)
    act_flattener = DictFlattener(env.action_space)
    envs = make_train_env(num_env)
    obss = envs.reset()
    actions = [{"red_fighter": act_space.sample(), 'blue_fighter': act_space.sample()} for
               _ in range(num_env)]
    each_fighter_obs_tuple = list(env.observation_space.spaces.items())[0]
    fighter_name = each_fighter_obs_tuple[0]
    offset = 0
    for obs_type in list(each_fighter_obs_tuple[1].spaces.items()):
        length = obs_type[1].shape[0]
        offset += length

    initial_state = actor.get_init_hidden_states(num_env)
    res = actor.get_action(actions, obss, initial_state[0])
    print(res[0], res[1])

    batch, seq, obsdims, actdims = 32, 4, 22, 4
    batch_pre_actions = torch.ones((batch, seq, actdims))
    batch_cur_obs = torch.ones((batch, seq, obsdims))
    batch_pre_gru_hidden = torch.ones((1, batch, 64))
    batch_cur_actions = torch.ones((batch, seq, actdims))
    output_logs, output_entropys = actor.bp_new_log_pi(batch_pre_actions, batch_cur_obs, batch_pre_gru_hidden, batch_cur_actions)
    print(output_logs.shape)



