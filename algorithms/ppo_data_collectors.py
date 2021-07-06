import copy
import numpy as np
import traceback
from .ppo_AC import ActorCritic
from .ppo_replaybuffer import ReplayBuffer
import time
import ray
import random
from ..envs.JSBSim.envs.self_play_env import JSBSimEnvSelfEnv


class BaseDataCollector(object):

    def __init__(self, args):
        self.env = args.env
        self.myown = ActorCritic(args).to(args.device)
        self.buffer = ReplayBuffer(args)
        self.cumulative_reward = None

        def signal_handler(signum, frame):
            raise Exception(f"Timeout on frame {frame}")

        # signal.signal(signal.SIGALRM, signal_handler)

    def print_message(self, **kwargs):
        pass

    def close(self):
        if self.env is None:
            return
        try:
            self.env.close()
        except:
            traceback.print_exc()

    def actions(self, blue_action):
        action = {'blue_fighter': blue_action, 'red_fighter': np.array([20., 18, 20., 11])}
        return action

    def collect_data(self, my_net_params, hyper_params):
        start_time = time.time()
        self.buffer.clear()
        episode_lists = []
        try:
            self.myown.load_state_dict(my_net_params)
            rollout_step, flag_rollout_abort, red_flag = 0, False, random.choice([True, False])
            # prevent the enemy from shooting with some probability
            total_episode = 0
            while not flag_rollout_abort:
                total_reward = 0
                my_cur_obs = self.env.reset()
                total_episode += 1
                my_cur_gru_h, pre_act = self.myown.get_init_hidden_state()
                while True:
                    my_cur_act, log_pi, my_next_gru_h, old_value = self.myown.get_action_value(pre_act, my_cur_obs, my_cur_gru_h)
                    try:
                        # signal.alarm(10)
                        actions = self.actions(self.myown.policy.act_flatten(my_cur_act))
                        my_next_obs, rewards, done, env_info = self.env.step(actions)
                        # signal.alarm(0)
                        total_reward += rewards['blue_reward']
                        rollout_step += 1
                    except:
                        # signal.alarm(0)
                        traceback.print_exc()
                        next_value, _ = self.myown.get_value(pre_act, my_cur_obs, my_cur_gru_h)
                        self.buffer.rollout_last_value = next_value
                        flag_rollout_abort = True
                        break
                    reward_weight = hyper_params['reward_hyper']
                    my_r = np.sum(reward_weight * rewards['blue_reward'])
                    self.buffer.add_sample(pre_act, my_cur_obs, my_cur_gru_h, my_cur_act, log_pi, old_value, my_r, done)
                    pre_act, my_cur_obs, my_cur_gru_h = my_cur_act, my_next_obs, my_next_gru_h
                    my_cur_obs = my_next_obs
                    if rollout_step >= self.buffer.buffer_size:
                        next_value, _ = self.myown.get_value(pre_act, my_cur_obs, my_cur_gru_h)
                        self.buffer.rollout_last_value = next_value
                        flag_rollout_abort = True
                    if done or flag_rollout_abort:
                        if done:
                            episode_lists.append(total_reward)
                            final_reward = rewards
                            # self.print_message(is_train=True, final_reward=final_reward, end_reason=env_info['end_reason'])
                        break


        except:
            traceback.print_exc()
            if rollout_step > 0:
                next_value, _ = self.myown.get_value(pre_act, my_cur_obs, my_cur_gru_h)
                self.buffer.rollout_last_value = next_value
            else:
                self.buffer.rollout_last_value = 0.
        time_elapsed = time.time() - start_time
        print(f"Collect data done, rollout steps {rollout_step}, time elapsed {time_elapsed}")
        status_code = 0 if rollout_step > 0 else 1
        self.cumulative_reward = np.mean(episode_lists)
        return status_code, self.buffer

    def collect_data_show(self, my_net_params, hyper_params):
        start_time = time.time()
        self.buffer.clear()
        episode_lists = []
        trajectory_list = []
        try:
            self.myown.load_state_dict(my_net_params)
            rollout_step, flag_rollout_abort, red_flag = 0, False, random.choice([True, False])
            # prevent the enemy from shooting with some probability
            total_episode = 0
            while not flag_rollout_abort:
                total_reward = 0
                my_cur_obs = self.env.reset()
                trajectory_list.append(self.env.render())
                total_episode += 1
                my_cur_gru_h, pre_act = self.myown.get_init_hidden_state()
                while True:
                    my_cur_act, log_pi, my_next_gru_h, old_value = self.myown.get_action_value(pre_act, my_cur_obs,
                                                                                               my_cur_gru_h)
                    try:
                        # signal.alarm(10)
                        actions = self.actions(self.myown.policy.act_flatten(my_cur_act))
                        my_next_obs, rewards, done, env_info = self.env.step(actions)
                        # signal.alarm(0)
                        trajectory_list.append(self.env.render())
                        total_reward += rewards['blue_reward']
                        rollout_step += 1
                    except:
                        # signal.alarm(0)
                        traceback.print_exc()
                        flag_rollout_abort = True
                        break
                    reward_weight = hyper_params['reward_hyper']
                    my_r = np.sum(reward_weight * rewards['blue_reward'])
                    pre_act, my_cur_obs, my_cur_gru_h = my_cur_act, my_next_obs, my_next_gru_h
                    my_cur_obs = my_next_obs
                    if rollout_step >= self.buffer.buffer_size:
                        next_value, _ = self.myown.get_value(pre_act, my_cur_obs, my_cur_gru_h)
                        self.buffer.rollout_last_value = next_value
                        flag_rollout_abort = True
                    if done or flag_rollout_abort:
                        if done:
                            episode_lists.append(total_reward)
                            final_reward = rewards
                            flag_rollout_abort = True
                            # self.print_message(is_train=True, final_reward=final_reward, end_reason=env_info['end_reason'])
                        break


        except:
            traceback.print_exc()
            if rollout_step > 0:
                next_value, _ = self.myown.get_value(pre_act, my_cur_obs, my_cur_gru_h)
                self.buffer.rollout_last_value = next_value
            else:
                self.buffer.rollout_last_value = 0.
        time_elapsed = time.time() - start_time
        print(f"Collect data done, rollout steps {rollout_step}, time elapsed {time_elapsed}")
        status_code = 0 if rollout_step > 0 else 1
        self.cumulative_reward = np.mean(episode_lists)
        return np.asarray(trajectory_list)


# @ray.remote(num_cpus=1)
class SelfPlayDataCollector(object):
    def __init__(self, args, flag_use_baseline=True):
        self.env = JSBSimEnvSelfEnv()
        self.buffer = ReplayBuffer(args)
        self.ego_policy, self.enm_policy = ActorCritic(args).to(args.device), ActorCritic(args).to(args.device)
        self.red_flag = True
        self.my_final_reward = None
        # record enemy's observation used to make decision.
        self.enm_history_info = None
        self.cumulative_reward = 0
        self.flag_use_baseline = flag_use_baseline
        self.rewards_list = []
        # statistics

    def _parse_obs(self, obs):
        # default: ego is blue fighter, enm is red fighter
        # TODO: CHECK here -> Finished
        obs_swapped = copy.deepcopy(obs)
        obs_swapped['blue_fighter'], obs_swapped['red_fighter'] = obs_swapped['red_fighter'], obs_swapped['blue_fighter']
        if self.red_flag:
            self.enm_cur_obs = obs
            return obs_swapped
        else:
            self.enm_cur_obs = obs_swapped
            return obs

    def _make_action(self, ego_cur_act, enm_cur_act):
        if self.flag_use_baseline:
            if self.red_flag:
                return {"red_fighter": ego_cur_act, 'blue_fighter': np.array([20., 18.6, 20., 0.])}
            else:
                return {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': ego_cur_act}
        if self.red_flag is True:
            return {'red_fighter': ego_cur_act, 'blue_fighter': enm_cur_act}
        else:
            return {'red_fighter': enm_cur_act, 'blue_fighter': ego_cur_act}

    def reset(self):
        obs = self.env.reset()
        enm_policy_gru, enm_pre_act_dict = self.enm_policy.get_init_hidden_state()
        self.enm_history_info = [enm_policy_gru, enm_pre_act_dict]
        self.my_final_reward = None
        return self._parse_obs(obs)

    def _choose_red_blue(self):
        if self.flag_use_baseline:
            self.red_flag = True
            return
        if self.red_flag:
            self.red_flag = False
        else:
            self.red_flag = True

    def step(self, ego_cur_act_array):
        enm_cur_act_dict, _, enm_cur_gru_h = self.enm_policy.get_action(self.enm_history_info[1], self.enm_cur_obs,
                                                                        self.enm_history_info[0])
        self.enm_history_info = [enm_cur_gru_h, enm_cur_act_dict]
        enm_cur_act_array = self.enm_policy.policy.act_flatten(enm_cur_act_dict)
        actions = self._make_action(ego_cur_act_array, enm_cur_act_array)
        next_obs, rewards, done, env_info = self.env.step(actions)
        my_next_obs = self._parse_obs(next_obs)
        my_rewards = rewards['red_reward' if self.red_flag else 'blue_reward']
        self.my_final_reward = rewards['red_final_reward' if self.red_flag else 'blue_final_reward']

        return my_next_obs, my_rewards, done, env_info

    def collect_data(self, ego_net_params, enm_net_params, hyper_params, agent_id=None):
        print(f'agent: {agent_id} starts to collect data')
        start_time = time.time()
        self.buffer.clear()
        rollout_step, flag_rollout_abort = 0, False
        total_reward, total_episode = 0, 0
        pooling_rewards = []
        try:
            self.enm_policy.load_state_dict(enm_net_params)
            self.ego_policy.load_state_dict(ego_net_params)
            while not flag_rollout_abort:
                self._choose_red_blue()
                ego_cur_obs = self.reset()
                total_episode += 1
                ego_cumulative_reward = 0
                ego_cur_gru_h, ego_pre_act_dict = self.ego_policy.get_init_hidden_state()
                while True:
                    res = self.ego_policy.get_action_value(ego_pre_act_dict, ego_cur_obs, ego_cur_gru_h)
                    ego_cur_act_dict, ego_log_pi, ego_next_gru_h, ego_old_value = res
                    ego_cur_act_array = self.ego_policy.policy.act_flatten(ego_cur_act_dict)
                    try:
                        ego_next_obs, ego_rewards, done, env_info = self.step(ego_cur_act_array)
                        rollout_step += 1
                    except:
                        traceback.print_exc()
                        ego_next_value, _ = self.ego_policy.get_value(ego_pre_act_dict, ego_cur_obs, ego_cur_gru_h)
                        self.buffer.rollout_last_value = ego_next_value
                        flag_rollout_abort = True
                        break
                    reward_scales = hyper_params['reward_hyper'][0]
                    # ego_cumulative_reward += np.sum(reward_scales * ego_rewards)
                    ego_cumulative_reward += ego_rewards
                    # TODO: easy for checking
                    ego_weighted_reward = ego_rewards + reward_scales * self.my_final_reward
                    self.buffer.add_sample(ego_pre_act_dict, ego_cur_obs['blue_fighter'], ego_cur_gru_h,
                                           ego_cur_act_dict, ego_log_pi, ego_old_value,
                                           ego_weighted_reward, done)
                    ego_pre_act_dict, ego_cur_gru_h = ego_cur_act_dict, ego_next_gru_h
                    ego_cur_obs = ego_next_obs

                    if rollout_step >= self.buffer.buffer_size:
                        ego_next_value, _ = self.ego_policy.get_value(ego_pre_act_dict, ego_cur_obs, ego_cur_gru_h)
                        self.buffer.rollout_last_value = ego_next_value
                        flag_rollout_abort = True
                    if done or flag_rollout_abort:
                        if done:
                            pooling_rewards.append(ego_cumulative_reward)
                        break
        except:
            traceback.print_exc()
            if rollout_step > 0:
                ego_next_value, _ = self.ego_policy.get_value(ego_pre_act_dict, ego_cur_obs, ego_cur_gru_h)
                self.buffer.rollout_last_value = ego_next_value
            else:
                self.buffer.rollout_last_value = 0.

        time_elapsed = time.time() - start_time
        print(f"Collect data done, rollout steps {rollout_step}, time elapsed {time_elapsed}")
        status_code = 0 if rollout_step > 0 else 1
        # self.cumulative_reward = np.asarray(pooling_rewards).mean()
        return status_code, self.buffer

    def evaluate_with_baseline(self, ego_net_params, enm_net_params, eval_num=1):
        start_time = time.time()
        rewards = 0
        self.ego_policy.load_state_dict(ego_net_params['model_state_dict'])
        self.enm_policy.load_state_dict(enm_net_params['model_state_dict'])

        for _ in range(eval_num):
            self._choose_red_blue()
            try:
                rollout_step = 0
                ego_cur_obs = self.reset()
                self.rewards_list = []
                ego_cur_gru_h, ego_pre_act_dict = self.ego_policy.get_init_hidden_state()
                while True:
                    self.env.render()
                    res = self.ego_policy.get_action_value(ego_pre_act_dict, ego_cur_obs, ego_cur_gru_h)
                    ego_cur_act_dict, ego_log_pi, ego_next_gru_h, ego_old_value = res
                    ego_cur_act_array = self.ego_policy.policy.act_flatten(ego_cur_act_dict)
                    try:
                        ego_next_obs, ego_rewards, done, env_info = self.step(ego_cur_act_array)
                        rollout_step += 1
                        rewards += ego_rewards
                        self.rewards_list.append(ego_rewards)
                    except:
                        traceback.print_exc()
                        break
                    ego_pre_act_dict, ego_cur_gru_h = ego_cur_act_dict, ego_next_gru_h
                    ego_cur_obs = ego_next_obs

                    if done:
                        # score = env_info['red_win' if self.red_flag else 'blue_win']
                        # eval_scores.append(score)
                        break
                    if rollout_step >= self.env.max_episode_steps:
                        break
            except:
                traceback.print_exc()

            np.save('kong_zhan', np.asarray(self.env.trajectory))
        rewards = rewards / eval_num
        print(rewards)
        return rewards

    def evaluate_data(self, ego_net_params, enm_net_params, ego_elo, enm_elo, eval_num, agent_id, elo_k=16.):
        start_time = time.time()
        eval_scores = []
        self.ego_policy.load_state_dict(ego_net_params)
        self.enm_policy.load_state_dict(enm_net_params)

        for _ in range(eval_num):
            self._choose_red_blue()
            self.red_flag = False
            try:
                rollout_step = 0
                ego_cur_obs = self.reset()
                ego_cur_gru_h, ego_pre_act_dict = self.ego_policy.get_init_hidden_state()
                while True:
                    res = self.ego_policy.get_action_value(ego_pre_act_dict, ego_cur_obs, ego_cur_gru_h)
                    ego_cur_act_dict, ego_log_pi, ego_next_gru_h, ego_old_value = res
                    ego_cur_act_array = self.ego_policy.policy.act_flatten(ego_cur_act_dict)
                    try:
                        ego_next_obs, ego_rewards, done, env_info = self.step(ego_cur_act_array)
                        rollout_step += 1
                    except:
                        traceback.print_exc()
                        break
                    ego_pre_act_dict, ego_cur_gru_h = ego_cur_act_dict, ego_next_gru_h
                    ego_cur_obs = ego_next_obs

                    if done:
                        score = env_info['red_win' if self.red_flag else 'blue_win']
                        eval_scores.append(score)
                        break
                    if rollout_step >= self.env.max_episode_steps:
                        break
            except:
                traceback.print_exc()

        if not eval_scores:
            elo_gain = 0.
        else:
            E_A = 1. / (1. + 10 ** ((enm_elo - ego_elo) / 400))
            elo_gain = np.mean(elo_k * (np.asarray(eval_scores) - E_A))
        time_elapsed = time.time() - start_time
        print(f"Evaluate done, time elapsed {time_elapsed} | agent: {agent_id}: elo_gain: {elo_gain}, {eval_scores}")
        status_code = 0 if eval_scores else 1
        return status_code, elo_gain





















