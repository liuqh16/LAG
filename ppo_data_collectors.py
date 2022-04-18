import numpy as np
import torch
import time
import gym
import ray
import logging
from envs import slimevolley
from ppo_buffer import PPOBuffer
from ppo_policy import PPOPolicy
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv

def _t2n(x):
    return x.detach().cpu().numpy()

def make_env(all_args):
    if all_args.env_name == "SingleCombat":
        env = SingleCombatEnv(all_args.scenario_name)
    elif all_args.env_name == "SingleControl":
        env = SingleControlEnv(all_args.scenario_name)
    elif all_args.env_name == "MultipleCombat":
        env = MultipleCombatEnv(all_args.scenario_name)
    else:
        logging.error("Can not support the " + all_args.env_name + "environment.")
        raise NotImplementedError
    # env.seed(all_args.seed)
    return env

class BaseDataCollector(object):
    def __init__(self, args) -> None:
        self.args = args
        self.env = make_env(args)
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.buffer_size = args.buffer_size
        self.num_agents = self.env.num_agents
        self.buffer = PPOBuffer(args, self.num_agents, self.env.observation_space, self.env.action_space)
        self.ego = PPOPolicy(args, self.env.observation_space, self.env.action_space)
    
    def collect_data(self, ego_params, hyper_params={}):
        self.buffer.clear()
        self.ego.restore_from_params(ego_params)
        obs = self.reset()
        self.buffer.obs[0] = obs.copy()
        for step in range(self.buffer_size):
            # 1. get actions
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = \
                self.ego.get_actions(self.buffer.obs[step],
                                     self.buffer.rnn_states_actor[step],
                                     self.buffer.rnn_states_critic[step],
                                     self.buffer.masks[step])
            values = _t2n(values)
            actions = _t2n(actions)
            action_log_probs = _t2n(action_log_probs)
            rnn_states_actor = _t2n(rnn_states_actor)
            rnn_states_critic = _t2n(rnn_states_critic)
            # 2. env step
            obs, rewards, dones, info = self.step(actions)
            if np.all(dones):
                obs = self.reset()
                rnn_states_actor = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
                rnn_states_critic = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
                masks = np.zeros((self.num_agents, 1), dtype=np.float32)
            else:
                masks = np.ones((self.num_agents, 1), dtype=np.float32)
            # 3. insert experience in buffer
            self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)
        status_code = 0 if step > 0 else 1
        last_value = self.ego.get_values(self.buffer.obs[-1], self.buffer.rnn_states_critic[-1], self.buffer.masks[-1])
        self.buffer.compute_returns(_t2n(last_value))
        return status_code, self.buffer
    
    @torch.no_grad()
    def evaluate_data(self, ego_params, hyper_params={}, ego_elo=0, enm_elo=0):
        self.ego.restore_from_params(ego_params)
        total_episode_rewards = []
        eval_scores = []
        max_episode_length = self.args.buffer_size
        episode_reward = 0
        eval_episodes = self.args.eval_episodes
        for _ in range(eval_episodes):
            obs = self.reset()
            rnn_states_actor = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
            masks = np.ones((self.num_agents, 1))
            step = 0
            while True:
                action, rnn_states_actor = self.ego.act(obs, rnn_states_actor, masks, deterministic=True)
                action = _t2n(action)
                rnn_states_actor = _t2n(rnn_states_actor)
                obs, reward, done, info = self.step(action)
                step += 1
                episode_reward += reward
                if np.all(done):
                    total_episode_rewards.append(episode_reward)
                    episode_reward = 0
                    score = info.get('score', 0.5)
                    eval_scores.append(score)
                    break
                if step >= max_episode_length:
                    break
        expected_score = 1 / (1 + 10 ** ((enm_elo - ego_elo) / 400))
        elo_gain = np.mean(eval_scores) - expected_score
        status_code = 0 if eval_scores else 1
        eval_info = info
        eval_info["episode_reward"] = np.mean(total_episode_rewards)
        return status_code, (elo_gain, eval_info)

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

class SelfPlayDataCollector(BaseDataCollector):

    def __init__(self, args):
        super().__init__(args)
        self.enm = PPOPolicy(args, self.env.observation_space, self.env.action_space)
        self.num_agents = self.num_agents // 2
        self.buffer = PPOBuffer(args, self.num_agents, self.env.observation_space, self.env.action_space)
    
    def collect_data(self, ego_params, enm_params, hyper_params={}):
        self.enm.restore_from_params(enm_params)
        return super().collect_data(ego_params, hyper_params)
    
    @torch.no_grad()
    def evaluate_data(self, ego_params, enm_params, hyper_params={}, ego_elo=0, enm_elo=0):
        self.enm.restore_from_params(enm_params)
        return super().evaluate_data(ego_params, hyper_params, ego_elo, enm_elo)
    
    def reset(self):
        self.enm_rnn_states_actor = np.zeros((self.num_agents, self.recurrent_hidden_layers, self.recurrent_hidden_size))
        self.enm_masks = np.ones((self.num_agents, 1))
        obs = super().reset()
        return self._parse_obs(obs)
    
    def step(self, action):
        enm_action, enm_rnn_states_actor = self.enm.act(self.enm_obs, self.enm_rnn_states_actor, self.enm_masks)
        enm_action = _t2n(enm_action)
        self.enm_rnn_states_actor = _t2n(enm_rnn_states_actor)
        actions = np.concatenate((action, enm_action), axis=0)
        n_obs, rewards, done, info = self.env.step(actions)
        ego_rewards = rewards[:self.num_agents, :]
        ego_done = done[:self.num_agents, :]
        enm_rewards = rewards[self.num_agents:, :]
        diff = (ego_rewards - enm_rewards).item()
        if diff > 50:
            info['score'] = 1
        elif diff < -50:
            info['score'] = 0
        else:
            info['score'] = 0.5
        return self._parse_obs(n_obs), ego_rewards, ego_done, info
    
    def _parse_obs(self, obs):
        self.enm_obs = obs[self.num_agents:, :]
        return obs[:self.num_agents, :]

@ray.remote(num_cpus=0.5)
class DataCollectorMix(object):
    def __init__(self, args) -> None:
        self.collector = None
        self.mode = None
        self.args = args
    
    def set_collector(self, mode):
        if self.mode == mode:
            return
        self.mode = mode
        if self.collector is not None:
            self.collector.close()
        if self.mode == 'base':
            self.collector = BaseDataCollector(self.args)
        elif self.mode == 'selfplay':
            self.collector = SelfPlayDataCollector(self.args)
        else:
            raise NotImplementedError

    def collect_data(self, ego_params, enm_params=None, hyper_params={}):
        if enm_params == None:
            mode = 'base'
        elif isinstance(enm_params, dict):
            mode = 'selfplay'
        else:
            raise NotImplementedError
        self.set_collector(mode)
        if self.mode == 'base':
            return self.collector.collect_data(ego_params, hyper_params)
        elif self.mode == 'selfplay':
            return self.collector.collect_data(ego_params, enm_params, hyper_params)
        else: 
            raise NotImplementedError
    
    def evaluate_data(self, ego_params, enm_params=None, hyper_params={}, ego_elo=0, enm_elo=0):
        if enm_params == None:
            mode = 'base'
        elif isinstance(enm_params, dict):
            mode = 'selfplay'
        else:
            raise NotImplementedError
        self.set_collector(mode)
        if self.mode == 'base':
            return self.collector.evaluate_data(ego_params, hyper_params, ego_elo, enm_elo)
        elif self.mode == 'selfplay':
            return self.collector.evaluate_data(ego_params, enm_params, hyper_params, ego_elo, enm_elo)
        else: 
            raise NotImplementedError