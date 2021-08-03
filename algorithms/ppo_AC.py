import numpy as np
import torch
import torch.nn as nn
from .ppo_actor import PolicyRnnMultiHead
from .ppo_critic import ValueRnnMultiHead


class ActorCritic(nn.Module):

    def __init__(self, args, agent_idx=None):
        super(ActorCritic, self).__init__()
        self.args = args
        self.num_policy_gru_hidden = args.policy_gru_config['hidden_size']
        self.policy = PolicyRnnMultiHead(args)
        self.value = ValueRnnMultiHead(args)
        self.agent_idx = agent_idx

    def forward(self,):
        pass

    def get_weight(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def save_model(self, rootpath='.', optimizer_state_dict=None, epoch_t=None, args=None):
        state_dict = {
            'model_state_dict': self.get_weight(),
            'optimizer_state_dict': optimizer_state_dict,
        }
        if epoch_t is not None:
            torch.save(state_dict, f"{rootpath}/models/agent{self.agent_idx}_history{epoch_t}.pt")
            torch.save(state_dict, f"{rootpath}/models/agent{self.agent_idx}_latest.pt")
        else:
            torch.save(state_dict, f"{rootpath}/models/agent{self.agent_idx}_latest.pt")

    @torch.no_grad()
    def get_action(self, pre_act_np, cur_obs_np, pre_gru_hidden_np):
        """
        Args:
            pre_act_np:             np.array[obs_space], len=num_envs
            cur_obs_np:             np.array[act_space], len=num_envs
            pre_gru_hidden_np:      np.array, shape=[num_layers, num_envs, N]
                If `self.args.flag_eval=True`, N = policy_hidden_size;
                Else, N = policy_hidden_size + value_hidden_size

        Returns:
            act_np:                 np.array[act_space], len=num_envs
            log_probs_np:           np.array, shape=[num_envs, act_dims]
            policy_cur_hidden_np:   np.array, shape=[num_layers, num_envs, policy_hidden_size]
        """
        if self.args.flag_eval:
            pre_gru_hidden_np = pre_gru_hidden_np
        else:
            pre_gru_hidden_np = pre_gru_hidden_np[:, :, :self.num_policy_gru_hidden]
        return self.policy.get_action(pre_act_np, cur_obs_np, pre_gru_hidden_np)

    @torch.no_grad()
    def get_action_value(self, pre_act_np, cur_obs_np, pre_gru_hidden_np):
        """
        Args:
            pre_act_np:             np.array[obs_space], len=num_envs
            cur_obs_np:             np.array[act_space], len=num_envs
            pre_gru_hidden_np:      np.array, shape=[num_layers, num_envs, policy_hidden_size + value_hidden_size]

        Returns:
            act_np:                 np.array[act_space], len=num_envs
            old_log_prob_np:        np.array, shape=[num_envs, act_dims]
            next_gru_hidden_np:     np.array, shape=[num_layers, num_envs, policy_hidden_size + value_hidden_size]
            value_np:               np.array, shape=[num_envs, ]
        """
        act_np, old_log_prob_np, next_policy_gru_hidden_np = self.get_action(pre_act_np, cur_obs_np, pre_gru_hidden_np)
        value_np, next_value_gru_hidden_np = self.get_value(pre_act_np, cur_obs_np, pre_gru_hidden_np)
        next_gru_hidden_np = np.concatenate([next_policy_gru_hidden_np, next_value_gru_hidden_np], axis=-1)
        return act_np, old_log_prob_np, next_gru_hidden_np, value_np

    def get_value(self, pre_act_np, cur_obs_np, pre_gru_hidden_np):
        """
        Args:
            pre_act_np:             np.array[obs_space], len=num_envs
            cur_obs_np:             np.array[act_space], len=num_envs
            pre_gru_hidden_np:      np.array, shape=[num_layers, num_envs, policy_hidden_size + value_hidden_size]

        Returns:
            value_np:               np.array, shape=[num_envs, ]
            value_cur_hidden_np:    np.array, shape=[num_layers, num_envs, value_hidden_size]
        """
        pre_gru_hidden_np = pre_gru_hidden_np[:, :, self.num_policy_gru_hidden:]
        return self.value.get_value(pre_act_np, cur_obs_np, pre_gru_hidden_np)

    def get_init_hidden_state(self, flag_eval=False, num_env=1):
        init_policy_gru_hidden_np, init_pre_act_np = self.policy.get_init_hidden_states(num_env)
        if flag_eval:
            return init_policy_gru_hidden_np, init_pre_act_np
        else:
            init_value_gru_hidden_np = self.value.get_init_hidden_states(num_env)
            init_gru_hidden_np = np.concatenate([init_policy_gru_hidden_np, init_value_gru_hidden_np], axis=-1)
            return init_gru_hidden_np, init_pre_act_np

    def bp_batch_new_log_pi(self, batch_pre_actions, batch_cur_observations, batch_pre_gru_hidden, batch_cur_actions):
        """
        Args:
            batch_pre_actions:              [batch_size, seq_len, act_dim]
            batch_cur_observations:         [batch_size, seq_len, obs_dim]
            batch_pre_gru_hidden:           [num_layers, batch_size, policy_hidden_size + value_hidden_size]
            batch_cur_act:                  [batch_size, seq_len, act_dim]

        Returns:
            output_logs:                    [batch_size * seq_len, act_dim]
            output_entropys:                [batch_size * seq_len, act_dim]
        """
        batch_pre_policy_gru_tensor = batch_pre_gru_hidden[:, :, :self.num_policy_gru_hidden]
        return self.policy.bp_new_log_pi(batch_pre_actions, batch_cur_observations, batch_pre_policy_gru_tensor, batch_cur_actions)

    def bp_batch_values(self, batch_pre_actions, batch_cur_observations, batch_pre_gru_hidden):
        """
        Args:
            batch_pre_actions:              [batch_size, seq_len, act_dim]
            batch_cur_observations:         [batch_size, seq_len, obs_dim]
            batch_pre_gru_hidden:           [num_layers, batch_size, policy_hidden_size + value_hidden_size]

        Returns:
            cur_values:                     [batch_size, seq_len, 1]
        """
        batch_pre_value_gru_tensor = batch_pre_gru_hidden[:, :, self.num_policy_gru_hidden:]
        return self.value.bp_new_value(batch_pre_actions, batch_cur_observations, batch_pre_value_gru_tensor)
