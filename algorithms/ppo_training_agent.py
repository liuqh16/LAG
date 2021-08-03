import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import traceback
import copy
from .ppo_AC import ActorCritic


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.agent_id, self.ac = None, ActorCritic(args)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=args.lr)
        self.init_lr = args.lr
        # 1. ppo
        self.discount_gamma, self.ppo_epoch = args.discount_gamma, args.ppo_epoch
        self.policy_clip, self.max_grad_norm = args.policy_clip, args.max_grad_norm
        self.entropy_weight, self.c = args.entropy_weight, args.tx_c
        # 2. used for saving data
        self.batch_size, self.seq_len = args.buffer_config['batch_size'], args.buffer_config['seq_len']
        self.total_chunk_data, self.num_total_chunk, self.num_envs_parallel = {}, None, None
        self.chunks_cur_obs, self.chunks_cur_act, self.chunks_cur_act, self.chunks_pred_values = [], [], [], []
        self.chunks_gru_h, self.chunks_returns, self.chunks_advantages, self.chunks_old_log_pis = [], [], [], []

    def get_state_dict(self):
        opt_state = copy.deepcopy(self.optimizer.state_dict())
        # move optimizer state to cpu
        for k, v in opt_state['state'].items():
            if torch.is_tensor(v):
                opt_state['state'][k] = v.cpu()
        for group in opt_state['param_groups']:
            for k, v in group.items():
                if torch.is_tensor(v):
                    group[k] = v.cpu()
        return {
            'model_state_dict': {k: v.cpu() for k, v in self.ac.state_dict().items()},
            'optimizer_state_dict': opt_state,
        }

    def load_state_dict(self, agent_params):
        self.ac.load_state_dict(agent_params['model_state_dict'])
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.init_lr)
        if agent_params['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(agent_params['optimizer_state_dict'])

    def update_agent(self, agent_id, agent_params, buffer_data_lists, hyper_params):
        print("updating")
        self.load_state_dict(agent_params)
        try:
            self.agent_id = agent_id
            self.split_episodes(buffer_data_lists)
            for idx_epoch in range(self.ppo_epoch):
                ret = self.buffer_random_shuffle()
                n_batches = (self.num_total_chunk - 1) // (self.batch_size * self.num_envs_parallel) + 1

                # cur_observations, cur_gru_hiddens, cur_actions, old_log_pis, old_values, returns, advantages
                for b in range(n_batches):
                    try:
                        start = b * self.batch_size * self.num_envs_parallel
                        end = min(self.num_total_chunk, (b + 1) * self.batch_size * self.num_envs_parallel)
                        pre_act_batch = ret[0][start:end]
                        cur_obs_batch, cur_gru_h_batch, cur_act_batch = ret[1][start:end], ret[2][start:end], ret[3][
                            start:end]
                        old_log_pis_batch, old_values_batch = ret[4][start:end], ret[5][start:end]
                        returns_batch, advantage_batch = ret[6][start:end], ret[7][start:end]

                        self._do_ppo_training(pre_act_batch, cur_obs_batch, cur_gru_h_batch, cur_act_batch, old_log_pis_batch,
                                              old_values_batch, returns_batch, advantage_batch, hyper_params)
                    except Exception as e:
                        print(f"Training failed for batch {b} because of {e}")
                        traceback.print_exc()
        except Exception as e:
            print(f"Training failed because of {e}")
            traceback.print_exc()

        for p in list(self.ac.parameters()):
            if torch.any(torch.isnan(p)):
                print("NaN occured after training, reverting changes")
                self.load_state_dict(agent_params)
                break
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return self.get_state_dict()

    def _do_ppo_training(self, pre_act_batch, cur_obs_batch, cur_gru_h_batch, cur_act_batch, old_log_pis_batch, old_values_batch,
                         returns_batch, advantage_batch, hyper_params):
        """
                cur_obs_batch:                numpy          [batch_size, seq_len, obs_shape]
                cur_gru_h_batch:              numpy          [batch_size, seq_len, (num_layer, hidden_size)]
                cur_act_batch:                numpy          [batch_size, seq_len, cur_act_shape]
                old_log_pis_batch:            numpy          [batch_size, seq_len, 3]
                reward_batch:                 numpy          [batch_size, seq_len, 1]
                next_obs_batch:               numpy          [batch_size, seq_len, obs_shape]
                terminal_batch:               numpy          [batch_size, seq_len, 1]
                returns_batch:                numpy          [batch_size, seq_len, 1]
                advantage_batch:              numpy          [batch_size, seq_len, 1]
        """
        """ 1. pre-process for updating """
        lr_scale, w_entropy_scale = hyper_params['ppo_hyper']
        cur_obs_b_shape = cur_obs_batch.shape[-1]
        # ------------------------- ------------------------------ history
        pre_act_b = torch.tensor(pre_act_batch, dtype=torch.float, device=self.device)
        cur_gru_hs = torch.tensor(cur_gru_h_batch, dtype=torch.float, device=self.device)[:, 0]  # [bs, nl, hs]
        cur_obs_b = torch.tensor(cur_obs_batch, dtype=torch.float, device=self.device)  # [batch_size, seq_len, obs_shape]
        cur_gru_h_b = cur_gru_hs.transpose(0, 1)  # [num_layer, batch_size, hidden_size]
        # ------------------------- ------------------------------ execute
        cur_act_b = torch.tensor(cur_act_batch, dtype=torch.float, device=self.device)  # [batch_size, seq_len, cur_act_shape]
        bs, sl = cur_act_b.shape[0:2]
        old_log_probs_b = torch.tensor(old_log_pis_batch, dtype=torch.float, device=self.device).view(bs*sl, -1)
        old_values_b = torch.tensor(old_values_batch, dtype=torch.float, device=self.device).view(bs*sl, -1)
        # ------------------------- ------------------------------   env
        returns_b = torch.tensor(returns_batch, dtype=torch.float, device=self.device).view(bs*sl, -1)
        advantage_b = torch.tensor(advantage_batch, dtype=torch.float, device=self.device).view(-1, 1)

        masks_tensor = torch.sign(torch.max(torch.abs(cur_obs_b.view(-1, cur_obs_b_shape)), dim=-1, keepdim=True)[0])
        """ 2. obtain the loss function   """
        new_log_probs, policy_entropys = self.ac.bp_batch_new_log_pi(pre_act_b, cur_obs_b, cur_gru_h_b, cur_act_b)
        new_predict_value = self.ac.bp_batch_values(pre_act_b, cur_obs_b, cur_gru_h_b).view(bs*sl, -1)
        assert new_log_probs.shape == old_log_probs_b.shape, (new_log_probs.shape, old_log_probs_b.shape)
        assert new_predict_value.shape == old_values_b.shape, (new_predict_value.shape, old_values_b.shape)
        # ---------------------------- ---------------------------------- policy loss
        # TODO: 1)
        ratio = torch.exp(new_log_probs - old_log_probs_b)
        # print('mask', masks_tensor.shape, 'ratio', ratio.shape)
        policy_surr1 = ratio * advantage_b * masks_tensor
        policy_surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * advantage_b * masks_tensor
        # TODO: 2)
        # policy_loss = -torch.max(torch.min(policy_surr1, policy_surr2),
        #                          self.c * advantage_b.clamp(-np.infty, 0) * masks_tensor).mean()
        policy_loss = -torch.min(policy_surr1, policy_surr2).mean()
        print('-------------------------------------------------------------------------------------')
        # print("policy_loss", policy_loss.item())

        # ---------------------------- ---------------------------------- value loss
        value_losses1 = (new_predict_value - returns_b).pow(2)
        # value_pred_clipped = old_values_b + (new_predict_value - old_values_b).clamp(
        #         #     -self.args.policy_clip, self.args.policy_clip)
        #         # value_losses2 = (value_pred_clipped - returns_b).pow(2)
        # value_loss = 0.5 * torch.min(value_losses1 * masks_tensor, value_losses2 * masks_tensor).mean()
        value_loss = 0.5 * (value_losses1 * masks_tensor).mean()
        # print("policy_loss", policy_loss.item(), "value_loss", value_loss.item())

        # ---------------------------- ---------------------------------- policy entropy loss
        policy_entropy_loss = -(policy_entropys * masks_tensor).mean()
        print("policy_loss", policy_loss.item(), "value_loss", value_loss.item(), "policy_entropy_loss", policy_entropy_loss.item())

        loss = lr_scale * (policy_loss + value_loss + w_entropy_scale * self.entropy_weight * policy_entropy_loss)
        print('total_loss', loss.item())
        """ 3. Optimize the loss function """
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

    def _compute_gae(self, cur_rewards, cur_values, terminals, last_value, gae_factor=0.95):
        gae, next_value = 0, last_value
        # print('cur_values', cur_values.shape)
        returns, advantages = np.zeros_like(cur_values), np.zeros_like(cur_values)
        for step in reversed(range(cur_rewards.shape[0])):
            td_delta = cur_rewards[step] + self.discount_gamma * (1. - terminals[step]) * next_value - cur_values[step]
            gae = self.discount_gamma * gae_factor * (1. - terminals[step]) * gae + td_delta
            advantages[step] = gae
            returns[step] = gae + cur_values[step]
            next_value = cur_values[step]
        # print('advantages', advantages.shape)
        return returns, advantages

    def _get_chunk_data(self, k, episode_start_idx, episode_length, episode_data, data_shape):
        chunk_start = k * self.seq_len
        chunk_end = min((k + 1) * self.seq_len, episode_length)
        b_states = np.zeros((self.seq_len, *data_shape))
        for t in range(chunk_end - chunk_start):
            b_states[t] = episode_data[chunk_start + episode_start_idx + t]
        return b_states

    def _split_episodes(self, data_class):
        trajectories = np.asarray(data_class.buffer, dtype=object)
        # -------------------------------------------------------------------------------------------------------------
        pre_actions = np.asarray(trajectories[:, 0].tolist(), dtype=np.float)        # 0. [total_steps, cur_act_shape]
        pre_act_shape = pre_actions.shape[1:]
        # print(pre_actions.shape)
        cur_observations = np.asarray(trajectories[:, 1].tolist(), dtype=np.float)   # 1. [total_steps, obs_shape]
        obs_shape = cur_observations.shape[1:]
        # print(cur_observations.shape)
        cur_gru_hiddens = np.asarray(trajectories[:, 2].tolist(), dtype=np.float)    # 2. [total_steps, gru_h_dims]
        gru_hidden_shape = cur_gru_hiddens.shape[1:]
        # print(cur_gru_hiddens.shape)
        # -------------------------------------------------------------------------------------------------------------
        cur_actions = np.asarray(trajectories[:, 3].tolist(), dtype=np.float)        # 3. [total_steps, cur_act_shape]
        cur_act_shape = cur_actions.shape[1:]
        old_log_pis = np.asarray(trajectories[:, 4].tolist(), dtype=np.float)        # 4. [total_steps, 2]
        log_pis_shape = old_log_pis.shape[1:]
        # print(old_log_pis.shape)
        pred_old_values = np.asarray(trajectories[:, 5].tolist(), dtype=np.float)    # 5. [total_steps, 1]
        values_shape = pred_old_values.shape[1:]
        # print(pred_old_values.shape)
        # -------------------------------------------------------------------------------------------------------------
        cur_rewards = np.asarray(trajectories[:, 6].tolist(), dtype=np.float)        # 6. [total_steps, 1]
        terminals = np.asarray(trajectories[:, 7].tolist(), dtype=np.float)          # 7. [total_steps, 1]
        # -------------------------------------------------------------------------------------------------------------
        # TODO: compute TD
        returns, advantages = self._compute_gae(cur_rewards, pred_old_values, terminals, data_class.rollout_last_value)
        # split episodes into chunks
        flag_splits = (np.where(terminals[:, 0] == True)[0] + 1).tolist()
        if len(flag_splits) == 0 or flag_splits[-1] != len(data_class.buffer):
            flag_splits.append(len(data_class.buffer))
        episode_start_idx = 0
        for episode_end_idx in flag_splits:
            episode_len = episode_end_idx - episode_start_idx
            for i in range((episode_len - 1) // data_class.seq_len + 1):
                # 0.  pre_act
                b_pre_act = self._get_chunk_data(i, episode_start_idx, episode_len, pre_actions, pre_act_shape)
                self.chunks_pre_act.append(b_pre_act)
                # 1.  cur_obs
                b_cur_obs = self._get_chunk_data(i, episode_start_idx, episode_len, cur_observations, obs_shape)
                self.chunks_cur_obs.append(b_cur_obs)
                # 2.  gru_h
                b_gru_h = self._get_chunk_data(i, episode_start_idx, episode_len, cur_gru_hiddens, gru_hidden_shape)
                self.chunks_gru_h.append(b_gru_h)
                # 3.  cur_act
                b_cur_act = self._get_chunk_data(i, episode_start_idx, episode_len, cur_actions, cur_act_shape)
                self.chunks_cur_act.append(b_cur_act)
                # 4.  old_log_pi
                b_old_log_pis = self._get_chunk_data(i, episode_start_idx, episode_len, old_log_pis, log_pis_shape)
                self.chunks_old_log_pis.append(b_old_log_pis)
                # 5.  old_value
                b_pred_value = self._get_chunk_data(i, episode_start_idx, episode_len, pred_old_values, values_shape)
                self.chunks_pred_values.append(b_pred_value)
                # 6.  returns
                b_rets = self._get_chunk_data(i, episode_start_idx, episode_len, returns, values_shape)
                self.chunks_returns.append(b_rets)
                # 7.  advantage
                b_advantage = self._get_chunk_data(i, episode_start_idx, episode_len, advantages, (1,))
                self.chunks_advantages.append(b_advantage)
            episode_start_idx = episode_end_idx

    def split_episodes(self, buffer_data_lists):
        self.total_chunk_data.clear()
        self.chunks_cur_obs, self.chunks_cur_act, self.chunks_pre_act, self.chunks_pred_values = [], [], [], []
        self.chunks_gru_h, self.chunks_returns, self.chunks_advantages, self.chunks_old_log_pis = [], [], [], []
        self.num_envs_parallel = len(buffer_data_lists)
        for buffer_data in buffer_data_lists:
            try:
                self._split_episodes(buffer_data)
            except Exception as e:
                print(f"Corrupt data chunk: {e}")
                traceback.print_exc()
        advs = np.asarray(self.chunks_advantages)
        self.total_chunk_data = {
            'pre_act': np.asarray(self.chunks_pre_act),
            'cur_obs': np.asarray(self.chunks_cur_obs),
            'gru_hidden': np.asarray(self.chunks_gru_h),
            'cur_act': np.asarray(self.chunks_cur_act),
            'old_log_pi': np.asarray(self.chunks_old_log_pis),
            'old_value': np.asarray(self.chunks_pred_values),
            'returns': np.asarray(self.chunks_returns),
            'advantages': (advs - advs.mean()) / (advs.std() + 1e-6)
        }

    def buffer_random_shuffle(self):
        pre_actions = self.total_chunk_data['pre_act']
        cur_observations = self.total_chunk_data['cur_obs']
        cur_gru_hiddens = self.total_chunk_data['gru_hidden']
        cur_actions = self.total_chunk_data['cur_act']
        old_log_pis = self.total_chunk_data['old_log_pi']
        old_values = self.total_chunk_data['old_value']
        returns = self.total_chunk_data['returns']
        advantages = self.total_chunk_data['advantages']
        self.num_total_chunk = len(cur_observations)
        random_idx = np.arange(self.num_total_chunk)
        np.random.shuffle(random_idx)

        pre_actions = pre_actions[random_idx]
        cur_observations = cur_observations[random_idx]
        cur_gru_hiddens = cur_gru_hiddens[random_idx]
        cur_actions = cur_actions[random_idx]
        old_log_pis = old_log_pis[random_idx]
        old_values = old_values[random_idx]
        returns = returns[random_idx]
        advantages = advantages[random_idx]
        return pre_actions, cur_observations, cur_gru_hiddens, cur_actions, old_log_pis, old_values, returns, advantages
