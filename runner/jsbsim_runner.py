import time
import numpy as np
import torch
import os

from .base_runner import Runner, ReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class JSBSimRunner(Runner):
    def __init__(self, config):
        super(JSBSimRunner, self).__init__(config)
        self.episode_length = self.all_args.episode_length
        self.total_num_steps = 0

    def load(self):
        assert len(self.envs.observation_space) == self.num_agents, f'{len(self.envs.observation_space)}, {self.num_agents}'
        obs_space = self.envs.observation_space[0]
        act_space = self.envs.action_space[0]

        # algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, obs_space, act_space, device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, obs_space, act_space)

        if self.model_dir is not None:
            self.restore()

    def save(self):
        save_dir = str(self.save_dir) + f"/{self.total_num_steps}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor, save_dir + "/actor.pth")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic, save_dir + "/critic.pth")

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):

            heading_turns_list = []
            for step in range(self.buffer_size):
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                self.insert(data)

                # Extra recorded information
                for info in infos:
                    if 'heading_turns' in info:
                        heading_turns_list.append(info['heading_turns'])

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                self.total_num_steps,
                                self.num_env_steps,
                                int(self.total_num_steps / (end - start))))

                done_splits = np.where(self.buffer._cast(self.buffer.dones) == True)[0] + 1
                sum_rewards = self.buffer._cast(self.buffer.rewards)[slice(*done_splits[[0, -1]])].sum()

                train_infos["average_episode_rewards"] = sum_rewards / (len(done_splits) - 1)
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    print("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states_actor[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        rnn_states_actor[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_actor.shape[3:]), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.envs.reset()
        eval_rnn_states = np.zeros((self.n_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        while total_episodes < self.eval_episodes:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                    np.concatenate(eval_rnn_states),
                                                                    deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions)
            eval_cumulative_rewards += eval_rewards

            eval_dones = np.all(eval_dones, axis=-1)
            total_episodes += np.sum(eval_dones)
            eval_episode_rewards += eval_cumulative_rewards[eval_dones == True].flatten().tolist()
            eval_cumulative_rewards[eval_dones == True] = 0
            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.array(eval_episode_rewards).mean()
        print("eval average episode rewards of agent: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/Recording_{self.experiment_name}.txt.acmi')
        while True:
            self.trainer.prep_rollout()
            render_actions, render_rnn_states = self.trainer.policy.act(np.concatenate(render_obs),
                                                                        np.concatenate(render_rnn_states),
                                                                        deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/Recording_{self.experiment_name}.txt.acmi')
            if render_dones.any():
                break

        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        print("render episode reward of agent: " + str(render_infos['render_episode_reward']))
