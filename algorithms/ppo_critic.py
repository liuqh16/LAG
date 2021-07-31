import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from .ppo_dictflatten import DictFlattener


class ValueRnnMultiHead(nn.Module):
    def __init__(self, args):
        super(ValueRnnMultiHead, self).__init__()
        self.args = args
        self.obs_space = args.observation_space
        self.act_space = args.action_space
        self.obs_flatten = DictFlattener(self.obs_space)
        self.act_flatten = DictFlattener(self.act_space)
        self.obs_dim = self.obs_flatten.size // len(self.obs_space.spaces.items())
        self.act_dim = self.act_flatten.size
        self.obs_slice = {}
        self.ego_name = None
        self._create_network()

    def _create_network(self):
        # 1. pre-process source observations.
        each_fighter_obs_tuple = list(self.obs_space.spaces.items())[0]  # ('blue_fighter', Dict(ego_info:Box(22,)))
        self.ego_name = each_fighter_obs_tuple[0]
        assert 'blue' in self.ego_name
        offset = 0
        for obs_type in list(each_fighter_obs_tuple[1].spaces.items()):
            length = obs_type[1].shape[0]
            offset += length
        self.obs_slice[self.ego_name] = (0, offset)
        self.ego_shape = each_fighter_obs_tuple[1].spaces['ego_info'].shape[0]
        self.pre_ego_net = nn.Sequential(nn.Linear(self.ego_shape + 4, self.args.value_ego[0]),
                                         nn.LayerNorm(self.args.value_ego[0]),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.args.value_ego[0], self.args.value_ego[1]),
                                         nn.ReLU(inplace=True))
        # 2. memory
        self.rnn_net = nn.GRU(input_size=self.args.value_ego[1],
                              num_layers=self.args.value_gru_config['num_layers'],
                              hidden_size=self.args.value_gru_config['hidden_size'], batch_first=True)
        # self.rnn_ln = nn.LayerNorm(self.args.value_gru_config['hidden_size'])
        # 3. Multi Head
        self.value_net = nn.Sequential(
            nn.Linear(self.args.value_gru_config['hidden_size'], self.args.value_mlp[0]),
            nn.LayerNorm(self.args.value_mlp[0]), nn.ReLU(inplace=True),
            nn.Linear(self.args.value_mlp[0], self.args.value_mlp[1]), nn.ReLU(inplace=True),
            nn.Linear(self.args.value_mlp[1], 1))

    def forward(self, cur_obs_ego, pi_pre_gru_h):
        # laser_tensor = self.pre_laser_net(cur_obs_laser)           # (batch, seq, laser_hidden_shape)
        ego_tensor = self.pre_ego_net(cur_obs_ego)                  # (batch, seq, ego_hidden_shape)
        # cur_gru_inputs = torch.cat([laser_tensor, ego_tensor], dim=-1)
        gru_out, gru_hidden = self.rnn_net(ego_tensor.contiguous(), pi_pre_gru_h.contiguous())
        # gru_out = self.rnn_ln(gru_out)                            # (batch, seq, gru_hidden_shape)
        value_tensor = self.value_net(gru_out)
        return value_tensor, gru_hidden

    def get_value(self, pre_act_lists, cur_obs_dict_lists, pre_gru_hidden_np):
        ego_nps = [cur_obs_dict[self.ego_name]['ego_info'] for cur_obs_dict in cur_obs_dict_lists]
        num_env = len(ego_nps)
        act_nps = [self.act_flatten(pre_act[self.ego_name]) / self.act_space['aileron'].n for pre_act in pre_act_lists]
        ego_np = np.concatenate([ego_nps, act_nps], axis=-1)

        cur_ego_tensor = torch.tensor(ego_np, dtype=torch.float, device=self.args.device).unsqueeze(1)
        pre_gru_hidden_tensor = torch.tensor(pre_gru_hidden_np, dtype=torch.float, device=self.args.device)
        value_tensor, value_cur_hidden_tensor = self.forward(cur_ego_tensor, pre_gru_hidden_tensor)
        value_np = value_tensor.detach().cpu().numpy()[0][0]
        # print(value_np)
        value_cur_hidden_np = value_cur_hidden_tensor.squeeze(dim=1).detach().cpu().numpy()
        return value_np, value_cur_hidden_np

    def bp_new_value(self, batch_pre_actions,  batch_cur_obs, batch_pre_gru_hidden, ):
        # batch_cur_gru_hidden:            tensor                 [num_layers, batch_size, gru_hidden_size]
        ego_obs = batch_cur_obs[:, :, slice(*self.obs_slice[self.ego_name])]
        ego_obs = torch.cat([ego_obs, batch_pre_actions / self.act_space['aileron'].n], dim=-1)
        cur_values, _ = self.forward(ego_obs, batch_pre_gru_hidden)
        return cur_values

    def get_init_hidden_states(self, num_env=1):
        cur_gru_hidden = np.zeros(
            [self.args.value_gru_config['num_layers'], num_env, self.args.value_gru_config['hidden_size']],
            dtype=np.float)
        return cur_gru_hidden


if __name__ == '__main__':
    import torch
    from algorithms.ppo_critic import ValueRnnMultiHead
    from algorithms.ppo_args import Config
    from envs.env_wrappers import SubprocVecEnv
    from envs.JSBSim.envs.selfplay_env import SelfPlayEnv

    def make_train_env(num_env):
        return SubprocVecEnv([SelfPlayEnv for _ in range(num_env)])

    num_env = 3
    envs = make_train_env(num_env)
    args = Config(env=envs)
    critic = ValueRnnMultiHead(args)

    obss = envs.reset()
    actions = [{'red_fighter': args.action_space.sample(),
                'blue_fighter': args.action_space.sample()} for _ in range(num_env)]
    initial_state = critic.get_init_hidden_states(num_env)
    res = critic.get_value(actions, obss, initial_state)
    print(res[0])
    print(res[1])

    batch, seq = args.buffer_config['batch_size'], args.buffer_config['seq_len']
    obsdims, actdims = critic.obs_dim, critic.act_dim
    batch_pre_actions = torch.ones((batch, seq, actdims))
    batch_cur_obs = torch.ones((batch, seq, obsdims))
    batch_pre_gru_hidden = torch.ones((1, batch, args.policy_gru_config['hidden_size']))
    batch_cur_actions = torch.ones((batch, seq, actdims))
    res = critic.bp_new_value(batch_cur_actions, batch_cur_obs, batch_pre_gru_hidden)
    print(res.shape)
