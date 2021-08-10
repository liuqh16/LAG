import torch
import numpy as np
from gym import spaces
import torch.nn as nn
from .ppo_dictflatten import DictFlattener


class ValueRnnMultiHead(nn.Module):
    def __init__(self, args):
        super(ValueRnnMultiHead, self).__init__()
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
        self.pre_ego_net = nn.Sequential(nn.Linear(self.obs_dim + self.act_dim, self.args.value_ego[0]),
                                         nn.LayerNorm(self.args.value_ego[0]),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(self.args.value_ego[0], self.args.value_ego[1]),
                                         nn.ReLU(inplace=True))
        # 2. memory
        self.rnn_net = nn.GRU(input_size=self.args.value_ego[1],
                              num_layers=self.args.value_gru_config['num_layers'],
                              hidden_size=self.args.value_gru_config['hidden_size'], batch_first=True)
        # 3. Multi Head
        self.value_net = nn.Sequential(
            nn.Linear(self.args.value_gru_config['hidden_size'], self.args.value_mlp[0]),
            nn.LayerNorm(self.args.value_mlp[0]), nn.ReLU(inplace=True),
            nn.Linear(self.args.value_mlp[0], self.args.value_mlp[1]), nn.ReLU(inplace=True),
            nn.Linear(self.args.value_mlp[1], 1))

    def forward(self, cur_obs_ego, pi_pre_gru_h):
        ego_tensor = self.pre_ego_net(cur_obs_ego)
        gru_out, gru_hidden = self.rnn_net(ego_tensor.contiguous(), pi_pre_gru_h.contiguous())
        value_tensor = self.value_net(gru_out)
        return value_tensor, gru_hidden

    def get_value(self, pre_act_np, cur_obs_np, pre_gru_hidden_np):
        """
        Args:
            pre_act_np:             np.array[obs_space], len=num_envs
            cur_obs_np:             np.array[act_space], len=num_envs
            pre_gru_hidden_np:      np.array, shape=[num_layers, num_envs, hidden_size]

        Returns:
            value_np:               np.array, shape=[num_envs, ]
            value_cur_hidden_np:    np.array, shape=[num_layers, num_envs, hidden_size]
        """
        ego_obs_nps = [self.obs_flatten(cur_obs) for cur_obs in cur_obs_np]
        ego_act_nps = [self.act_flatten(pre_act) / self.act_norm for pre_act in pre_act_np]
        ego_nps = np.concatenate([ego_obs_nps, ego_act_nps], axis=-1)

        cur_ego_tensor = torch.tensor(ego_nps, dtype=torch.float, device=self.args.device).unsqueeze(1)     # [num_envs, 1, obs_dim + act_dim]
        pre_gru_hidden_tensor = torch.tensor(pre_gru_hidden_np, dtype=torch.float, device=self.args.device) # [num_layers, num_envs, hidden_size]

        value_tensor, value_cur_hidden_tensor = self.forward(cur_ego_tensor, pre_gru_hidden_tensor)
        value_np = value_tensor.detach().cpu().numpy().squeeze()
        value_cur_hidden_np = value_cur_hidden_tensor.squeeze(dim=1).detach().cpu().numpy()
        return value_np, value_cur_hidden_np

    def bp_new_value(self, batch_pre_act, batch_cur_obs, batch_pre_gru_hidden):
        """
        Args:
            batch_pre_act:          [batch_size, seq_len, act_dim]
            batch_cur_obs:          [batch_size, seq_len, obs_dim]
            batch_pre_gru_hidden:   [num_layers, batch_size, hidden_size]

        Returns:
            cur_values:             [batch_size, seq_len, 1]
        """
        ego_nps = torch.cat([batch_cur_obs, batch_pre_act / self.act_norm], dim=-1)
        cur_values, _ = self.forward(ego_nps, batch_pre_gru_hidden)
        return cur_values

    def get_init_hidden_states(self, num_env=1):
        init_gru_hidden_np = np.zeros([self.args.value_gru_config['num_layers'],
                                       num_env, self.args.value_gru_config['hidden_size']], dtype=np.float)
        return init_gru_hidden_np


if __name__ == '__main__':
    import torch
    from algorithms.ppo_critic import ValueRnnMultiHead
    from algorithms.ppo_args import Config
    from envs.env_wrappers import SubprocVecEnv
    from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv

    def make_train_env(num_env):
        return SubprocVecEnv([SingleCombatEnv for _ in range(num_env)])

    num_env = 3
    envs = make_train_env(num_env)
    args = Config(env=envs)
    critic = ValueRnnMultiHead(args)

    obss = envs.reset()
    acts = [{'red_fighter': args.action_space.sample(),
                'blue_fighter': args.action_space.sample()} for _ in range(num_env)]
    ego_acts = [act['red_fighter'] for act in acts]
    ego_obss = [obs['red_fighter'] for obs in obss]
    initial_state = critic.get_init_hidden_states(num_env)
    res = critic.get_value(ego_acts, ego_obss, initial_state)
    print(res[0].shape)
    print(res[1].shape)

    batch, seq = args.buffer_config['batch_size'], args.buffer_config['seq_len']            # (512, 8)
    obsdims, actdims = critic.obs_dim, critic.act_dim                                       # (22, 4)
    batch_pre_actions = torch.ones((batch, seq, actdims))                                   # (512, 8, 4)
    batch_cur_obs = torch.ones((batch, seq, obsdims))                                       # (512, 8, 22)
    batch_pre_gru_hidden = torch.ones((1, batch, args.policy_gru_config['hidden_size']))    # (1, 512, 128)
    batch_cur_actions = torch.ones((batch, seq, actdims))                                   # (512, 8, 4)
    res = critic.bp_new_value(batch_cur_actions, batch_cur_obs, batch_pre_gru_hidden)       # (512, 8, 1)
    print(res.shape)
