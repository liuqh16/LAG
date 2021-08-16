import numpy as np
import torch
import torch.nn as nn
from .ppo_policy import PPOPolicy
from ..utils.buffer import ReplayBuffer
from ..utils.util import check


class PPOTrainer():
    def __init__(self, args, policy: PPOPolicy, device=torch.device("cpu")):

        self.policy = policy
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # ppo config
        self.ppo_epoch = args.ppo_epoch
        self.clip_param = args.clip_param
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        # rnn configs
        self.use_recurrent_policy = args.use_recurrent_policy

    def ppo_update(self, sample):

        obs_batch, rnn_states_actor_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, old_action_log_probs_batch, advantages_batch = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = \
            self.policy.evaluate_actions(obs_batch,
                                         rnn_states_actor_batch, 
                                         rnn_states_critic_batch, 
                                         actions_batch)

        # Obtain the loss function
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()
        loss = action_loss + value_loss * self.value_loss_coef - dist_entropy * self.entropy_coef

        # Optimize the loss function
        self.policy.optimizer.zero_grad()
        loss.backward()
        if self.use_max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        return loss, action_loss, value_loss, dist_entropy

    def train(self, buffer: ReplayBuffer, turn_on=True):
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self.use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, ratio \
                    = self.ppo_update(sample, turn_on)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
