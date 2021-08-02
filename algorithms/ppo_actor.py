import pdb
import torch
import numpy as np
from gym import spaces
import torch.nn as nn
from collections import OrderedDict
from torch.distributions import Categorical
from .ppo_dictflatten import DictFlattener


class PolicyRnnMultiHead(nn.Module):
    def __init__(self, args):
        super(PolicyRnnMultiHead, self).__init__()
        self.args = args
        # create ego observation & action space
        all_observation_space = args.observation_space  # NOTE: contains ego&enm obs
        ego_name = list(all_observation_space.spaces.keys())[0]

        self.obs_space = all_observation_space[ego_name]
        self.obs_flatten = DictFlattener(self.obs_space)
        self.obs_dim = self.obs_flatten.size

        self.act_space = args.action_space
        self.act_flatten = DictFlattener(self.act_space)
        self.act_dim = self.act_flatten.size
        self.act_norm = max([space.n for space in self.act_space.spaces.values() if isinstance(space, spaces.Discrete)])

        self._create_network()
        self.to(args.device)

    def _create_network(self):
        # 1. pre-process source observations.
        self.pre_ego_net = nn.Sequential(nn.Linear(self.obs_dim + self.act_dim, self.args.policy_ego[0]),
                                         nn.LayerNorm(self.args.policy_ego[0]),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.args.policy_ego[0], self.args.policy_ego[1]),
                                         nn.ReLU(inplace=True))
        # 2. memory
        self.rnn_net = nn.GRU(input_size=self.args.policy_ego[1],
                              num_layers=self.args.policy_gru_config['num_layers'],
                              hidden_size=self.args.policy_gru_config['hidden_size'], batch_first=True)
        # 3. Multi Head
        output_multi_head = {}
        for act_name, act_space in self.act_space.spaces.items():
            channel_output = nn.Sequential(
                nn.Linear(self.args.policy_gru_config['hidden_size'], self.args.policy_act_mlp[0]),
                nn.LayerNorm(self.args.policy_act_mlp[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.args.policy_act_mlp[0], self.args.policy_act_mlp[1]),
                nn.ReLU(inplace=True),
                nn.Linear(self.args.policy_act_mlp[1], act_space.n))
            output_multi_head[act_name] = channel_output
        self.output_multi_head = nn.ModuleDict(output_multi_head)

    def forward(self, cur_obs_ego, pi_pre_gru_h):
        ego_tensor = self.pre_ego_net(cur_obs_ego)
        gru_out, gru_hidden = self.rnn_net(ego_tensor.contiguous(), pi_pre_gru_h.contiguous())
        output_dists = {}
        for act_name, act_space in self.act_space.spaces.items():
            channel_act_prob = torch.softmax(self.output_multi_head[act_name](gru_out).reshape(-1, act_space.n), dim=-1)
            channel_act_dist = Categorical(channel_act_prob)
            output_dists[act_name] = channel_act_dist
        return output_dists, gru_hidden

    def get_action(self, pre_act_np, cur_obs_np, pre_gru_hidden_np):
        """
        Args:
            pre_act_np:             np.array[obs_space], len=num_envs
            cur_obs_np:             np.array[act_space], len=num_envs
            pre_gru_hidden_np:      np.array, shape=[num_layers, num_envs, hidden_size]

        Returns:
            act_np:                 np.array[act_space], len=num_envs
            log_probs_np:           np.array, shape=[num_envs, act_dims]
            policy_cur_hidden_np:   np.array, shape=[num_layers, num_envs, hidden_size]
        """
        ego_obs_nps = [self.obs_flatten(cur_obs) for cur_obs in cur_obs_np]
        ego_act_nps = [self.act_flatten(pre_act) / self.act_norm for pre_act in pre_act_np]
        ego_nps = np.concatenate([ego_obs_nps, ego_act_nps], axis=-1)

        cur_ego_tensor = torch.tensor(ego_nps, dtype=torch.float, device=self.args.device).unsqueeze(1)
        pre_gru_hidden_tensor = torch.tensor(pre_gru_hidden_np, dtype=torch.float, device=self.args.device)

        output_dists, gru_hidden = self.forward(cur_ego_tensor, pre_gru_hidden_tensor)
        actions, log_probs = [], []
        for act_name in output_dists.keys():
            channel_dist = output_dists[act_name]
            channel_act = channel_dist.sample() if not self.args.flag_eval else channel_dist.logits.max(dim=-1)[1]
            channel_act_log_prob = channel_dist.log_prob(channel_act)
            actions.append(channel_act.unsqueeze(-1))                       # [num_envs, 1]
            log_probs.append(channel_act_log_prob.unsqueeze(-1))            # [num_envs, 1]

        actions_np = torch.cat(actions, dim=-1).cpu().detach().numpy()      # [num_envs, act_dim]
        log_probs_np = torch.cat(log_probs, dim=-1).cpu().detach().numpy()  # [num_envs, act_dim]
        policy_cur_hidden_np = gru_hidden.detach().cpu().numpy()            # [num_layers, num_envs, hidden_size]
        act_np = np.asarray([self.act_flatten.inv(action) for action in actions_np], dtype=object)
        return act_np, log_probs_np, policy_cur_hidden_np

    def bp_new_log_pi(self, batch_pre_act, batch_cur_obs, batch_pre_gru_hidden, batch_cur_act, no_grad=False):
        """
        Args:
            batch_pre_act:          [batch_size, seq_len, act_dim]
            batch_cur_obs:          [batch_size, seq_len, obs_dim]
            batch_pre_gru_hidden:   [num_layers, batch_size, hidden_size]
            batch_cur_act:          [batch_size, seq_len, act_dim]

        Returns:
            output_logs:            [batch_size * seq_len, act_dim]
            output_entropys:        [batch_size * seq_len, act_dim]
        """
        with torch.set_grad_enabled(not no_grad):
            ego_obs = torch.cat([batch_cur_obs, batch_pre_act / self.act_norm], dim=-1)
            output_dists, _  = self.forward(ego_obs, batch_pre_gru_hidden)
            output_logs, output_entropys = [], []
            for act_idx, act_name in enumerate(output_dists.keys()):
                channel_act = batch_cur_act[:, :, act_idx].long().reshape(-1, 1)
                log_prob = output_dists[act_name].log_prob(channel_act.squeeze(-1)).unsqueeze(-1)
                output_logs.append(log_prob)
                output_entropys.append(output_dists[act_name].entropy().unsqueeze(-1))
            output_logs, output_entropys = torch.cat(output_logs, dim=-1), torch.cat(output_entropys, dim=-1)
        return output_logs, output_entropys

    def get_init_hidden_states(self, num_env=1):
        init_gru_hidden_np = np.zeros([self.args.policy_gru_config['num_layers'],
                                    num_env, self.args.policy_gru_config['hidden_size']], dtype=np.float)
        init_pre_act_np = np.asarray([self.act_flatten.inv(np.zeros(self.act_dim)) for _ in range(num_env)], dtype=object)
        return init_gru_hidden_np, init_pre_act_np


if __name__ == '__main__':
    import torch
    from algorithms.ppo_actor import PolicyRnnMultiHead
    from algorithms.ppo_args import Config
    from envs.env_wrappers import SubprocVecEnv
    from envs.JSBSim.envs.selfplay_env import SelfPlayEnv

    def make_train_env(num_env):
        return SubprocVecEnv([SelfPlayEnv for _ in range(num_env)])

    num_env = 3
    envs = make_train_env(num_env)
    args = Config(env=envs)
    actor = PolicyRnnMultiHead(args)

    obss = envs.reset()
    acts = [{'red_fighter': args.action_space.sample(),
                'blue_fighter': args.action_space.sample()} for _ in range(num_env)]
    ego_acts = [act['red_fighter'] for act in acts]
    ego_obss = [obs['red_fighter'] for obs in obss]
    init_hidden_states = actor.get_init_hidden_states(num_env)
    next_actions, log_probs, next_hidden_states = actor.get_action(ego_acts, ego_obss, init_hidden_states[0])
    print(next_actions)
    print(log_probs)
    print(next_hidden_states.shape)

    batch, seq = args.buffer_config['batch_size'], args.buffer_config['seq_len']
    obsdims, actdims = actor.obs_dim, actor.act_dim
    batch_pre_actions = torch.ones((batch, seq, actdims))
    batch_cur_obs = torch.ones((batch, seq, obsdims))
    batch_pre_gru_hidden = torch.ones((1, batch, args.policy_gru_config['hidden_size']))
    batch_cur_actions = torch.ones((batch, seq, actdims))
    output_logs, output_entropys = actor.bp_new_log_pi(batch_pre_actions, batch_cur_obs, batch_pre_gru_hidden, batch_cur_actions)
    print(output_logs.shape)
