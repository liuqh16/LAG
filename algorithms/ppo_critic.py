import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from envs.JSBSim.tasks.selfplay_task import act_space, obs_space
from .ppo_dictflatten import DictFlattener, obs_slice


class ValueRnnMultiHead(nn.Module):
    def __init__(self, args):
        super(ValueRnnMultiHead, self).__init__()
        self.args = args
        self.obs_flatten = DictFlattener(obs_space)
        self.act_flatten = DictFlattener(act_space)
        self._create_network()
        self.to(args.device)

    def _create_network(self, ):
        # 1. pre-process source observations.
        # 1.1
        # self.laser_shape = obs_space['blue_car']['laser_info'].shape[0]
        # self.pre_laser_net = nn.Sequential(nn.Linear(self.laser_shape, self.args.value_laser[0]),
        #                                    nn.LayerNorm(self.args.value_laser[0]), nn.ReLU(inplace=True),
        #                                    nn.Linear(self.args.value_laser[0], self.args.value_laser[1]), nn.ReLU(inplace=True))
        # 1.2
        self.ego_shape = obs_space['blue_fighter']['ego_info'].shape[0]
        # self.key_shape = obs_space['blue_car']['key_info'].shape[0]
        self.pre_ego_net = nn.Sequential(nn.Linear(self.ego_shape + 4, self.args.value_ego[0]),
                                         nn.LayerNorm(self.args.value_ego[0]), nn.ReLU(inplace=True),
                                         nn.Linear(self.args.value_ego[0], self.args.value_ego[1]), nn.ReLU(inplace=True))
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
        # laser_tensor = self.pre_laser_net(cur_obs_laser)            # (batch, seq, laser_hidden_shape)
        ego_tensor = self.pre_ego_net(cur_obs_ego)                  # (batch, seq, ego_hidden_shape)
        # cur_gru_inputs = torch.cat([laser_tensor, ego_tensor], dim=-1)
        gru_out, gru_hidden = self.rnn_net(ego_tensor.contiguous(), pi_pre_gru_h.contiguous())
        # gru_out = self.rnn_ln(gru_out)                              # (batch, seq, gru_hidden_shape)
        # 1.
        value_tensor = self.value_net(gru_out)
        return value_tensor, gru_hidden

    def get_value(self, pre_act_np, cur_obs_dict, pre_gru_hidden_np):
        # laser_np = cur_obs_dict['blue_car']['laser_info']
        ego_np = cur_obs_dict['blue_fighter']['ego_info']
        # key_np = cur_obs_dict['blue_car']['key_info']
        ego_np = np.concatenate([ego_np, self.act_flatten(pre_act_np)/act_space['aileron'].n], axis=-1)
        # cur_laser_tensor = torch.tensor(laser_np, dtype=torch.float, device=self.args.device).view(1, 1, -1)
        cur_ego_tensor = torch.tensor(ego_np, dtype=torch.float, device=self.args.device).view(1, 1, -1)
        pre_gru_hidden_tensor = torch.tensor(pre_gru_hidden_np, dtype=torch.float, device=self.args.device).unsqueeze(dim=1)
        value_tensor, value_cur_hidden_tensor = self.forward(cur_ego_tensor, pre_gru_hidden_tensor)
        value_np = value_tensor.detach().cpu().numpy()[0][0]
        # print(value_np)
        value_cur_hidden_np = value_cur_hidden_tensor.squeeze(dim=1).detach().cpu().numpy()
        return value_np, value_cur_hidden_np

    def bp_new_value(self, batch_pre_actions,  batch_cur_obs, batch_pre_gru_hidden, ):
        # batch_cur_gru_hidden:            tensor                 [num_layers, batch_size, gru_hidden_size]
        batch_size, seq_len = batch_cur_obs.shape[0:2]
        # laser_obs = batch_cur_obs[:, :, slice(*obs_slice['laser_shape'])]
        ego_obs = batch_cur_obs[:, :, slice(*obs_slice['ego_shape'])]
        # key_obs = batch_cur_obs[:, :, slice(*obs_slice['key_shape'])]
        ego_obs = torch.cat([ego_obs, batch_pre_actions/act_space['aileron'].n], dim=-1)
        cur_values, _ = self.forward(ego_obs, batch_pre_gru_hidden)
        return cur_values

    def get_init_hidden_states(self):
        cur_gru_hidden = np.zeros([self.args.value_gru_config['num_layers'], self.args.value_gru_config['hidden_size']],
                                  dtype=np.float)
        return cur_gru_hidden


if __name__ == '__main__':
    class Config(object):
        def __init__(self):
            self.value_laser = [32, 32]
            self.value_ego = [32, 32]
            self.value_gru_config = {'num_layers': 1, 'hidden_size': 64}
            self.value_mlp = [32, 32]
            self.flag_eval = False
            self.device = torch.device('cpu')

    pi = ValueRnnMultiHead(Config())
    obs = obs_space.sample()
    init_pre_act = OrderedDict({'aileron': 5, 'elevator': 5, 'rudder': 5, 'throttle': 0})
    print(pi.get_value(init_pre_act, obs, pi.get_init_hidden_states())[0])
    cur_obs = torch.ones((12, 4, 39))
    batch_pre_gru_hidden = torch.ones((1, 12, 64))
    batch_cur_actions = torch.ones((12, 4, 4))
    res = pi.bp_new_value(batch_cur_actions, cur_obs, batch_pre_gru_hidden)
    print(res.shape)











