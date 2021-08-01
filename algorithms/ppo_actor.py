import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import Categorical
from .ppo_dictflatten import DictFlattener


class PolicyRnnMultiHead(nn.Module):
    def __init__(self, args):
        super(PolicyRnnMultiHead, self).__init__()
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
        self.to(args.device)

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
        self.pre_ego_net = nn.Sequential(nn.Linear(self.ego_shape + self.act_dim, self.args.policy_ego[0]),
                                         nn.LayerNorm(self.args.policy_ego[0]),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.args.policy_ego[0], self.args.policy_ego[1]),
                                         nn.ReLU(inplace=True))
        # 2. memory
        self.rnn_net = nn.GRU(input_size=self.args.policy_ego[1],
                              num_layers=self.args.policy_gru_config['num_layers'],
                              hidden_size=self.args.policy_gru_config['hidden_size'], batch_first=True)
        # self.rnn_ln = nn.LayerNorm(self.args.policy_gru_config['hidden_size'])
        # 3. Multi Head
        output_multi_head = {}
        for act_channel, act_space in self.act_space.spaces.items():
            single_output = nn.Sequential(
                nn.Linear(self.args.policy_gru_config['hidden_size'], self.args.policy_act_mlp[0]),
                nn.LayerNorm(self.args.policy_act_mlp[0]), nn.ReLU(inplace=True),
                nn.Linear(self.args.policy_act_mlp[0], self.args.policy_act_mlp[1]), nn.ReLU(inplace=True),
                nn.Linear(self.args.policy_act_mlp[1], act_space.n))
            output_multi_head[act_channel] = single_output

        self.output_multi_head = nn.ModuleDict(output_multi_head)

    def forward(self, cur_obs_ego, pi_pre_gru_h):
        import pdb; pdb.set_trace()
        # laser_tensor = self.pre_laser_net(cur_obs_laser)            # (batch, seq, laser_hidden_shape)
        ego_tensor = self.pre_ego_net(cur_obs_ego)                    # (batch, seq, ego_hidden_shape)
        # cur_gru_inputs = torch.cat([laser_tensor, ego_tensor], dim=-1)
        gru_out, gru_hidden = self.rnn_net(ego_tensor.contiguous(), pi_pre_gru_h.contiguous())
        # gru_out = self.rnn_ln(gru_out)                              # (batch, seq, gru_hidden_shape)
        output_dists = {}
        for act_channel, act_space in self.act_space.spaces.items():
            single_action_prob = torch.softmax(self.output_multi_head[act_channel](gru_out).reshape(-1, act_space.n), dim=-1)
            single_action_dist = Categorical(single_action_prob)
            output_dists[act_channel] = single_action_dist

        return output_dists, gru_hidden

    def get_action(self, pre_act_lists, cur_obs_dict_lists, pre_gru_hidden_np):
        ego_nps = [cur_obs_dict[self.ego_name]['ego_info'] for cur_obs_dict in cur_obs_dict_lists]
        num_env = len(ego_nps)
        act_nps = [self.act_flatten(pre_act[self.ego_name]) / self.act_space['aileron'].n for pre_act in pre_act_lists]
        ego_nps, act_nps = np.array(ego_nps), np.array(act_nps)
        ego_np = np.concatenate([ego_nps, act_nps], axis=-1)

        cur_ego_tensor = torch.tensor(ego_np, dtype=torch.float, device=self.args.device).unsqueeze(1)
        pre_gru_hidden_tensor = torch.tensor(pre_gru_hidden_np, dtype=torch.float, device=self.args.device)
        import pdb; pdb.set_trace()
        output_dists, gru_hidden = self.forward(cur_ego_tensor, pre_gru_hidden_tensor)
        actions, log_probs = [], []
        for act_channel in self.act_space.spaces.keys():
            single_dist = output_dists[act_channel]                                      # (batch, )
            single_act = single_dist.sample() if not self.args.flag_eval else single_dist.logits.max(dim=-1)[1]
            single_act_log_prob = single_dist.log_prob(single_act)                       # (batch, )
            actions.append(single_act.unsqueeze(-1))
            log_probs.append(single_act_log_prob.unsqueeze(-1))
        actions, log_probs = torch.cat(actions, dim=-1), torch.cat(log_probs, dim=-1)    # (batch, self.act_dim)
        action_for_envs = []
        for i in range(num_env):
            act_for_each_env = {}
            for j, act_type in enumerate(self.act_space.spaces.items()):
                act_for_each_env[act_type[0]] = actions[i, j].cpu().detach().numpy()
            action_for_envs.append(OrderedDict(act_for_each_env))
        return action_for_envs, log_probs.cpu().detach().numpy(), gru_hidden.detach().cpu().numpy()

    def bp_new_log_pi(self, batch_pre_actions, batch_cur_obs, batch_pre_gru_hidden, batch_cur_actions, no_grad=False):
        # batch_cur_gru_hidden:            tensor                 [num_layers, batch_size, gru_hidden_size]
        with torch.set_grad_enabled(not no_grad):
            ego_obs = batch_cur_obs[:, :, slice(*self.obs_slice[self.ego_name])]
            ego_obs = torch.cat([ego_obs, batch_pre_actions / self.act_space['aileron'].n], dim=-1)
            res = self.forward(ego_obs, batch_pre_gru_hidden)
            output_dists, _ = res
            output_logs, output_entropys = [], []
            for act_id, act_type in enumerate(self.act_space.spaces.items()):
                single_act = batch_cur_actions[:, :, act_id].long().reshape(-1, 1)
                log_prob = output_dists[act_type[0]].log_prob(single_act.squeeze(-1)).unsqueeze(-1)
                output_logs.append(log_prob)
                output_entropys.append(output_dists[act_type[0]].entropy().unsqueeze(-1))
            output_logs, output_entropys = torch.cat(output_logs, dim=-1), torch.cat(output_entropys, dim=-1)
        return output_logs, output_entropys

    def get_init_hidden_states(self, num_env=1):
        cur_gru_hidden = np.zeros(
            [self.args.policy_gru_config['num_layers'], num_env, self.args.policy_gru_config['hidden_size']],
            dtype=np.float)
        init_pre_act = [OrderedDict({'aileron': 20, 'elevator': 18, 'rudder': 20, 'throttle': 11}) for _ in range(num_env)]
        return cur_gru_hidden, init_pre_act


if __name__ == '__main__':
    import torch
    from algorithms.ppo_actor import PolicyRnnMultiHead
    from algorithms.ppo_args import Config
    from envs.env_wrappers import SubprocVecEnv
    from envs.JSBSim.envs.selfplay_env import SelfPlayEnv

    def make_train_env(num_env):
        return SubprocVecEnv([SelfPlayEnv for _ in range(num_env)])

    num_env = 2
    envs = make_train_env(num_env)
    args = Config(env=envs)
    actor = PolicyRnnMultiHead(args)

    obss = envs.reset()
    actions = [{'red_fighter': args.action_space.sample(),
                'blue_fighter': args.action_space.sample()} for _ in range(num_env)]
    initial_state = actor.get_init_hidden_states(num_env)
    res = actor.get_action(actions, obss, initial_state[0])
    print(res[0])
    print(res[1])

    batch, seq = args.buffer_config['batch_size'], args.buffer_config['seq_len']
    obsdims, actdims = actor.obs_dim, actor.act_dim
    batch_pre_actions = torch.ones((batch, seq, actdims))
    batch_cur_obs = torch.ones((batch, seq, obsdims))
    batch_pre_gru_hidden = torch.ones((1, batch, args.policy_gru_config['hidden_size']))
    batch_cur_actions = torch.ones((batch, seq, actdims))
    output_logs, output_entropys = actor.bp_new_log_pi(batch_pre_actions, batch_cur_obs, batch_pre_gru_hidden, batch_cur_actions)
    print(output_logs.shape)
