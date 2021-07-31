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

    def save_model(self, optimizer_state_dict=None, epoch_t=None, args=None):
        state_dict = {
            'model_state_dict': self.get_weight(),
            'optimizer_state_dict': optimizer_state_dict,
        }
        if epoch_t is not None:
            torch.save(state_dict, f"./models/{args.version}/agent{self.agent_idx}_history{epoch_t}.pt")
            torch.save(state_dict, f"./models/{args.version}/agent{self.agent_idx}_latest.pt")
        else:
            torch.save(state_dict, f"./models/{args.version}/agent{self.agent_idx}_latest.pt")

    @torch.no_grad()
    def get_action(self, pre_act_np, cur_obs_np, cur_gru_hidden_np):
        if self.args.flag_eval:
            cur_policy_gru_np = cur_gru_hidden_np
        else:
            cur_policy_gru_np = cur_gru_hidden_np[:, :self.num_policy_gru_hidden]
        return self.policy.get_action(pre_act_np, cur_obs_np, cur_policy_gru_np)

    @torch.no_grad()
    def get_action_value(self, pre_act_np, cur_obs_np, cur_gru_hidden_np):
        cur_action, old_log_prob, next_policy_gru_hidden_np = self.get_action(pre_act_np, cur_obs_np, cur_gru_hidden_np)
        value, next_value_gru_hidden_np = self.get_value(pre_act_np, cur_obs_np, cur_gru_hidden_np)
        next_gru_hidden_np = np.concatenate([next_policy_gru_hidden_np, next_value_gru_hidden_np], axis=-1)
        return cur_action, old_log_prob, next_gru_hidden_np, value

    def get_value(self, pre_act_np, cur_obs_np, cur_gru_hidden_np):
        cur_value_gru_np = cur_gru_hidden_np[:, self.num_policy_gru_hidden:]
        return self.value.get_value(pre_act_np, cur_obs_np, cur_value_gru_np)

    def get_init_hidden_state(self, flag_eval=False):
        policy_gru, pre_act = self.policy.get_init_hidden_states()
        if flag_eval:
            return policy_gru, pre_act
        else:
            value_gru = self.value.get_init_hidden_states()
            init_gru_hidden = np.concatenate([policy_gru, value_gru], axis=-1)
            return init_gru_hidden, pre_act

    def bp_batch_new_log_pi(self, batch_pre_actions, batch_cur_observations, batch_cur_gru_hidden, batch_cur_actions):
        """
        batch_cur_observations:          tensor                 [batch_size, seq_len, obs_shape]
        batch_cur_gru_hidden:            tensor                 [num_layers, batch_size, gru_hidden_size * 2]
        batch_cur_actions:               tensor                 [batch_size, seq_len, cur_act_shape]
        return:                          tensor                 [batch_size * seq_len, 2]
        """
        cur_policy_gru_tensor = batch_cur_gru_hidden[:, :, :self.num_policy_gru_hidden]
        return self.policy.bp_new_log_pi(batch_pre_actions, batch_cur_observations, cur_policy_gru_tensor, batch_cur_actions)

    def bp_batch_values(self, batch_pre_actions, batch_cur_observations, batch_pre_gru_hidden):
        """
        batch_cur_observations:          tensor                 [batch_size, seq_len, obs_shape]
        batch_cur_gru_hidden:            tensor                 [num_layers, batch_size, gru_hidden_size * 2]
        return:                          tensor                 [batch_size * seq_len, 1]
        """
        pre_value_gru_tensor = batch_pre_gru_hidden[:, :, self.num_policy_gru_hidden:]
        return self.value.bp_new_value(batch_pre_actions, batch_cur_observations, pre_value_gru_tensor)


