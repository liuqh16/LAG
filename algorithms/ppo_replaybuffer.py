import numpy as np
from collections import deque
from envs.JSBSim.tasks.selfplay_task import obs_space, act_space
from .ppo_dictflatten import DictFlattener


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.obs_flatten = DictFlattener(obs_space['blue_fighter'])
        self.act_flatten = DictFlattener(act_space)
        self.buffer_size, self.seq_len = self.args.buffer_config['buffer_size'], self.args.buffer_config['seq_len']
        self.buffer = deque(maxlen=self.buffer_size)
        self.rollout_last_value = None

    def clear(self):
        self.buffer.clear()

    def add_sample(self, pre_act, cur_obs, pre_gru_h, cur_act, old_log_pi, pred_old_value, cur_reward, done):
        """
        cur_obs:                            O_{t}                    history         0.
        cur_gru_h:                          h_{t}                    history         1.
        pre_act:                            a_{t-1}                  history         2.
        ----------------------------------------------------------------------------------------------
        cur_act:                            a_{t}                    execute         3.
        old_log_pi:                     log_prob_a                   execute         4.
        pred_old_value:                 V(o_t, h_t, a_{t-1})         execute         5.
        -----------------------------------------------------------------------------------------------
        cur_reward:                         r_t                        env           6.
        next_obs:                          O_{t+1}                     env           7.
        done:                                                          env           8.
        pre_act_tmp = np.zeros(self.act_dims, dtype=np.float)     pre_act_tmp[int(pre_act.item())] = 1.
        pre_act = pre_act_tmp
        """
        cur_obs, pre_act, cur_act = self.obs_flatten(cur_obs), self.act_flatten(pre_act), self.act_flatten(cur_act)
        self.buffer.append([pre_act, cur_obs, pre_gru_h, cur_act, old_log_pi, pred_old_value, [np.clip(cur_reward, -20., 20.)],
                            [done]])
