import time
import pdb
import numpy as np
import traceback
from collections import OrderedDict

from numpy.core.fromnumeric import size
from .ppo_AC import ActorCritic
from .ppo_replaybuffer import ReplayBuffer


class SelfPlayDataCollector(object):
    def __init__(self, args, flag_use_baseline=True):
        self.env = args.env
        self.num_envs = getattr(self.env, 'num_envs', 1)  # type: int
        self.buffers = [ReplayBuffer(args) for _ in range(self.num_envs)]
        self.ego_policy, self.enm_policy = ActorCritic(args).to(args.device), ActorCritic(args).to(args.device)
        self.red_flag = np.random.choice([True, False], size=self.num_envs)
        self.flag_use_baseline = flag_use_baseline

    def _parse_obs(self, obs: np.ndarray):
        """Parse obs_dict_list into ego+enmy obs
        """
        # default: ego is blue fighter, enm is red fighter
        ego_cur_obss = []
        enm_cur_obss = []
        for i in range(self.num_envs):
            if self.red_flag[i]:
                ego_cur_obss.append(obs[i]['red_fighter'])
                enm_cur_obss.append(obs[i]['blue_fighter'])
            else:
                ego_cur_obss.append(obs[i]['blue_fighter'])
                enm_cur_obss.append(obs[i]['red_fighter'])
        ego_cur_obss = np.asarray(ego_cur_obss, dtype=object)
        enm_cur_obss = np.asarray(enm_cur_obss, dtype=object)
        return ego_cur_obss, enm_cur_obss

    def _make_action(self, ego_cur_act_list: list, enm_cur_act_list: list):
        """Return action_dict_list for env input
        """
        acts = []
        for i in range(self.num_envs):
            ego_cur_act_array = self.ego_policy.policy.act_flatten(ego_cur_act_list[i])
            enm_cur_act_array = self.enm_policy.policy.act_flatten(enm_cur_act_list[i])
            if self.flag_use_baseline:
                enm_cur_act_array = np.array([20., 18.6, 20., 0.])
            if self.red_flag[i]:
                act = {'red_fighter': ego_cur_act_array, 'blue_fighter': enm_cur_act_array}
            else:
                act = {'red_fighter': enm_cur_act_array, 'blue_fighter': ego_cur_act_array}
            acts.append(act)
        return np.asarray(acts, dtype=object)

    def _parse_rewards(self, rewards: np.ndarray):
        """Parse reward_dict_list into ego+enmy reward
        """
        ego_rewards = np.zeros(self.num_envs)
        enm_rewards = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            if self.red_flag[i]:
                ego_rewards[i] = rewards[i]['red_fighter']
                enm_rewards[i] = rewards[i]['blue_fighter']
            else:
                ego_rewards[i] = rewards[i]['blue_fighter']
                enm_rewards[i] = rewards[i]['red_fighter']
        return ego_rewards, enm_rewards

    def warmup(self):
        # clear buffer
        for i in range(self.num_envs):
            self.buffers[i].clear()
        self.red_flag = np.random.choice([True, False], size=self.num_envs)

    def step(self, ego_cur_acts, enm_cur_acts):
        acts = self._make_action(ego_cur_acts, enm_cur_acts)
        next_obss, rewards, dones, env_infos = self.env.step(acts)
        ego_next_obss, enm_next_obss = self._parse_obs(next_obss)
        ego_rewards, enm_rewards = self._parse_rewards(rewards)
        return ego_next_obss, enm_next_obss, ego_rewards, enm_rewards, dones, env_infos

    def collect_data(self, ego_net_params, enm_net_params, hyper_params=None, agent_id=None):
        print(f'agent: {agent_id} starts to collect data')
        start_time = time.time()
        # load policy
        self.enm_policy.load_state_dict(enm_net_params)
        self.ego_policy.load_state_dict(ego_net_params)
        rollout_step, flag_rollout_abort = 0, False
        # start collecting
        self.warmup()
        ego_cur_obss, enm_cur_obss = self._parse_obs(self.env.reset())
        ego_cur_gru, ego_pre_acts = self.ego_policy.get_init_hidden_state(num_env=self.num_envs)
        enm_cur_gru, enm_pre_acts = self.enm_policy.get_init_hidden_state(flag_eval=True, num_env=self.num_envs)
        while not flag_rollout_abort:
            ego_cur_acts, ego_log_pi, ego_next_gru, ego_old_values = self.ego_policy.get_action_value(ego_pre_acts, ego_cur_obss, ego_cur_gru)
            enm_cur_acts, _, enm_next_gru = self.enm_policy.get_action(enm_pre_acts, enm_cur_obss, enm_cur_gru)
            ego_next_obss, enm_next_obss, ego_rewards, enm_rewards, dones, env_infos = self.step(ego_cur_acts, enm_cur_acts)
            rollout_step += 1
            # ego_rewards += hyper_params['reward_hyper'][0] * final_reward
            for i in range(self.num_envs):
                self.buffers[i].add_sample(ego_pre_acts[i],
                                           ego_cur_obss[i],
                                           ego_cur_gru[:, i, :],
                                           ego_cur_acts[i],
                                           ego_log_pi[i],
                                           ego_old_values[i],
                                           ego_rewards[i],
                                           dones[i])
            ego_pre_acts, ego_cur_obss, ego_cur_gru = ego_cur_acts, ego_next_obss, ego_next_gru
            enm_pre_acts, enm_cur_obss, enm_cur_gru = enm_cur_acts, enm_next_obss, enm_next_gru
            if rollout_step >= self.buffers[0].buffer_size:
                ego_next_values, _ = self.ego_policy.get_value(ego_pre_acts, ego_cur_obss, ego_cur_gru)
                for i in range(self.num_envs):
                    self.buffers[i].rollout_last_value = ego_next_values[i]
                flag_rollout_abort = True
            # deal with env.reset situations
            self.red_flag[dones] = np.random.choice([True, False], size=np.sum(dones))
            ego_cur_gru[:, dones, :], ego_pre_acts[dones] = self.ego_policy.get_init_hidden_state(num_env=np.sum(dones))
            enm_cur_gru[:, dones, :], enm_pre_acts[dones] = self.enm_policy.get_init_hidden_state(flag_eval=True, num_env=np.sum(dones))

        time_elapsed = time.time() - start_time
        print(f"Collect data done, rollout steps {rollout_step * self.num_envs}, time elapsed {time_elapsed}")
        status_code = 0 if rollout_step > 0 else 1
        return status_code, self.buffers

    def evaluate_with_baseline(self, ego_net_params, enm_net_params, eval_num=1):
        print('#####################################################################################')
        print('\nStart evaluating...')
        self.ego_policy.load_state_dict(ego_net_params['model_state_dict'])
        ego_cur_obss, _ = self._parse_obs(self.env.reset())
        ego_cur_gru, ego_pre_acts = self.ego_policy.get_init_hidden_state(num_env=self.num_envs)
        cumulative_rewards = np.zeros(self.num_envs)
        episode_rewards = []
        total_episodes = 0
        while total_episodes < eval_num:
            ego_cur_acts, _, ego_next_gru, _ = self.ego_policy.get_action_value(ego_pre_acts, ego_cur_obss, ego_cur_gru)
            ego_next_obss, _, ego_rewards, _, dones, env_infos = self.step(ego_cur_acts, ego_cur_acts.copy())
            ego_pre_acts, ego_cur_obss, ego_cur_gru = ego_cur_acts, ego_next_obss, ego_next_gru
            cumulative_rewards += ego_rewards
            # deal with env.reset situations
            total_episodes += np.sum(dones)
            episode_rewards += cumulative_rewards[dones].tolist()
            cumulative_rewards[dones] = 0
            self.red_flag[dones] = np.random.choice([True, False], size=np.sum(dones))
            ego_cur_gru[:, dones, :], ego_pre_acts[dones] = self.ego_policy.get_init_hidden_state(num_env=np.sum(dones))

        average_rewards = np.mean(episode_rewards)
        print(f"Average episode_reward = {average_rewards}")
        print('#####################################################################################')
        return average_rewards

    def collect_data_once(self, ego_net_params, enm_net_params):
        assert self.num_envs == 1
        self.ego_policy.load_state_dict(ego_net_params)
        trajectory_list, ego_cumulative_reward =[], 0
        ego_cur_gru, ego_pre_act = self.ego_policy.get_init_hidden_state(num_env=1)
        # start rendering
        ego_cur_obs, _ = self._parse_obs(np.expand_dims(np.asarray(self.env.reset()), axis=0))
        trajectory_list.append(self.env.render())
        while True:
            ego_cur_act, _, ego_next_gru = self.ego_policy.get_action(ego_pre_act, ego_cur_obs, ego_cur_gru)
            acts = self._make_action(ego_cur_act, ego_cur_act.copy())
            next_obs, reward, done, env_info = self.env.step(acts[0])
            ego_next_obs, _ = self._parse_obs(np.expand_dims(np.asarray(next_obs), axis=0))
            ego_reward, _ = self._parse_rewards(np.expand_dims(np.asarray(reward), axis=0))
            trajectory_list.append(self.env.render())
            ego_cumulative_reward += ego_reward[0]
            ego_pre_act, ego_cur_gru, ego_cur_obs = ego_cur_act, ego_next_gru, ego_next_obs
            if done:
                break
        print(f"Ego({'red' if self.red_flag else 'blue'}) accumulate reward = {ego_cumulative_reward:.2f}")
        return np.asarray(trajectory_list)
