import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import Categorical
from envs.collections_task.self_play_task import act_space, obs_space
from .ppo_dictflatten import DictFlattener, obs_slice


class PolicyRnnMultiHead(nn.Module):
    def __init__(self, args):
        super(PolicyRnnMultiHead, self).__init__()
        self.args = args
        self.obs_flatten = DictFlattener(obs_space)
        self.act_flatten = DictFlattener(act_space)
        self._create_network()

    def _create_network(self, ):
        # 1. pre-process source observations.
        # 1.1

        # 1.2
        self.ego_shape = obs_space['blue_fighter']['ego_info'].shape[0]
        self.pre_ego_net = nn.Sequential(nn.Linear(self.ego_shape + 4, self.args.policy_ego[0]),
                                         nn.LayerNorm(self.args.policy_ego[0]), nn.ReLU(inplace=True),
                                         nn.Linear(self.args.policy_ego[0], self.args.policy_ego[1]),
                                         nn.ReLU(inplace=True))
        # 2. memory
        self.rnn_net = nn.GRU(input_size=self.args.policy_ego[1],
                              num_layers=self.args.policy_gru_config['num_layers'],
                              hidden_size=self.args.policy_gru_config['hidden_size'], batch_first=True)
        # self.rnn_ln = nn.LayerNorm(self.args.policy_gru_config['hidden_size'])
        # 3. Multi Head
        self.aileron_last_logits = nn.Sequential(
            nn.Linear(self.args.policy_gru_config['hidden_size'], self.args.policy_act_mlp[0]),
            nn.LayerNorm(self.args.policy_act_mlp[0]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[0], self.args.policy_act_mlp[1]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[1], act_space['aileron'].n))

        self.elevator_last_logits = nn.Sequential(
            nn.Linear(self.args.policy_gru_config['hidden_size'], self.args.policy_act_mlp[0]),
            nn.LayerNorm(self.args.policy_act_mlp[0]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[0], self.args.policy_act_mlp[1]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[1], act_space['elevator'].n))

        self.rudder_last_logits = nn.Sequential(
            nn.Linear(self.args.policy_gru_config['hidden_size'], self.args.policy_act_mlp[0]),
            nn.LayerNorm(self.args.policy_act_mlp[0]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[0], self.args.policy_act_mlp[1]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[1], act_space['rudder'].n))

        self.throttle_last_logits = nn.Sequential(
            nn.Linear(self.args.policy_gru_config['hidden_size'], self.args.policy_act_mlp[0]),
            nn.LayerNorm(self.args.policy_act_mlp[0]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[0], self.args.policy_act_mlp[1]), nn.ReLU(inplace=True),
            nn.Linear(self.args.policy_act_mlp[1], act_space['throttle'].n))

    def forward(self, cur_obs_ego, pi_pre_gru_h):
        # laser_tensor = self.pre_laser_net(cur_obs_laser)            # (batch, seq, laser_hidden_shape)
        ego_tensor = self.pre_ego_net(cur_obs_ego)                  # (batch, seq, ego_hidden_shape)
        # cur_gru_inputs = torch.cat([laser_tensor, ego_tensor], dim=-1)
        gru_out, gru_hidden = self.rnn_net(ego_tensor.contiguous(), pi_pre_gru_h.contiguous())
        # gru_out = self.rnn_ln(gru_out)                              # (batch, seq, gru_hidden_shape)
        # 1.
        action_aileron_prob = torch.softmax(self.aileron_last_logits(gru_out).reshape(-1, act_space['aileron'].n), dim=-1)
        action_aileron_dist = Categorical(action_aileron_prob)
        action_elevator_prob = torch.softmax(self.elevator_last_logits(gru_out).reshape(-1, act_space['elevator'].n), dim=-1)
        action_elevator_dist = Categorical(action_elevator_prob)
        action_rudder_prob = torch.softmax(self.rudder_last_logits(gru_out).reshape(-1, act_space['rudder'].n), dim=-1)
        action_rudder_dist = Categorical(action_rudder_prob)
        action_throttle_prob = torch.softmax(self.throttle_last_logits(gru_out).reshape(-1, act_space['throttle'].n), dim=-1)
        action_throttle_dist = Categorical(action_throttle_prob)
        return action_aileron_dist, action_elevator_dist, action_rudder_dist, action_throttle_dist, gru_hidden

    def get_action(self, pre_act, cur_obs_dict, pre_gru_hidden_np):
        # print(cur_obs_dict)
        ego_np = cur_obs_dict['blue_fighter']['ego_info']
        ego_np = np.concatenate([ego_np, self.act_flatten(pre_act)/act_space['aileron'].n], axis=-1)
        cur_ego_tensor = torch.tensor(ego_np, dtype=torch.float, device=self.args.device).view(1, 1, -1)
        pre_gru_hidden_tensor = torch.tensor(pre_gru_hidden_np, dtype=torch.float, device=self.args.device).unsqueeze(dim=1)
        res = self.forward(cur_ego_tensor, pre_gru_hidden_tensor)
        action_aileron_dist, action_elevator_dist, action_rudder_dist, action_throttle_dist, gru_hidden = res
        aileron = action_aileron_dist.sample() if not self.args.flag_eval else action_aileron_dist.logits.max(dim=-1)[1]
        aileron_log_probs = action_aileron_dist.log_prob(aileron)
        elevator = action_elevator_dist.sample() if not self.args.flag_eval else action_elevator_dist.logits.max(dim=-1)[1]
        elevator_log_probs = action_elevator_dist.log_prob(elevator)
        rudder = action_rudder_dist.sample() if not self.args.flag_eval else action_rudder_dist.logits.max(dim=-1)[1]
        rudder_log_probs = action_rudder_dist.log_prob(rudder)
        throttle = action_throttle_dist.sample() if not self.args.flag_eval else action_throttle_dist.logits.max(dim=-1)[1]
        throttle_log_probs = action_throttle_dist.log_prob(throttle)
        cur_action = OrderedDict({
            'aileron': aileron.item(),
            'elevator': elevator.item(),
            'rudder': rudder.item(),
            'throttle': throttle.item(),
        })
        log_probs = torch.cat([aileron_log_probs, elevator_log_probs, rudder_log_probs, throttle_log_probs], dim=-1).cpu().detach().squeeze().numpy()
        policy_cur_hidden_np = gru_hidden.squeeze(dim=1).detach().cpu().numpy()
        return cur_action, log_probs, policy_cur_hidden_np

    def bp_new_log_pi(self, batch_pre_actions, batch_cur_obs, batch_pre_gru_hidden, batch_cur_actions, no_grad=False):
        # batch_cur_gru_hidden:            tensor                 [num_layers, batch_size, gru_hidden_size]
        with torch.set_grad_enabled(not no_grad):
            ego_obs = batch_cur_obs[:, :, slice(*obs_slice['ego_shape'])]
            ego_obs = torch.cat([ego_obs, batch_pre_actions/act_space['aileron'].n], dim=-1)
            res = self.forward(ego_obs, batch_pre_gru_hidden)
            action_aileron_dist, action_elevator_dist, action_rudder_dist, action_throttle_dist, _ = res
            cmd_aileron = batch_cur_actions[:, :, 0].long().reshape(-1, 1)
            cmd_elevator = batch_cur_actions[:, :, 1].long().reshape(-1, 1)
            cmd_rudder = batch_cur_actions[:, :, 2].long().reshape(-1, 1)
            cmd_throttle = batch_cur_actions[:, :, 3].long().reshape(-1, 1)
            aileron_log_prob = action_aileron_dist.log_prob(cmd_aileron.squeeze(-1)).unsqueeze(-1)
            elevator_log_prob = action_elevator_dist.log_prob(cmd_elevator.squeeze(-1)).unsqueeze(-1)
            rudder_log_prob = action_rudder_dist.log_prob(cmd_rudder.squeeze(-1)).unsqueeze(-1)
            throttle_log_prob = action_throttle_dist.log_prob(cmd_throttle.squeeze(-1)).unsqueeze(-1)
            new_log_prob = torch.cat([aileron_log_prob, elevator_log_prob, rudder_log_prob, throttle_log_prob], dim=-1)
            policy_entropies = torch.cat(
                [action_aileron_dist.entropy(), action_elevator_dist.entropy(), action_rudder_dist.entropy(),
                 action_throttle_dist.entropy()], dim=-1)
        return new_log_prob, policy_entropies

    def get_init_hidden_states(self):
        cur_gru_hidden = np.zeros([self.args.policy_gru_config['num_layers'], self.args.policy_gru_config['hidden_size']], dtype=np.float)
        init_pre_act = OrderedDict({'aileron': 20, 'elevator': 18, 'rudder': 20, 'throttle': 11})
        return cur_gru_hidden, init_pre_act


if __name__ == '__main__':
    class Config(object):
        def __init__(self):
            self.act_dims = 10
            self.policy_ego = [32, 32]
            self.policy_gru_config = {'num_layers': 1, 'hidden_size': 64}
            self.policy_act_mlp = [32, 32]
            self.flag_eval = False
            self.device = torch.device('cpu')

    pi = PolicyRnnMultiHead(Config())
    obs = obs_space.sample()
    print(obs)
    print(pi.get_action(pi.get_init_hidden_states()[1], obs, pi.get_init_hidden_states()[0])[0])
    cur_obs = torch.ones((12, 4, 39))
    batch_pre_gru_hidden = torch.ones((1, 12, 64))
    batch_cur_actions = torch.ones((12, 4, 4))
    res = pi.bp_new_log_pi(batch_cur_actions, cur_obs, batch_pre_gru_hidden, batch_cur_actions)
    print(res[0].shape)











