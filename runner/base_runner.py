import os
import sys
import wandb
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.utils.buffer import ReplayBuffer
import logging

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.current_episode = 0
        self.eval_render_mode = config.get('eval_render_mode', None)
        self.render_mode = config.get('render_mode', "txt")


        
        # Tacview render obj
        self.tacview = None
        if self.eval_render_mode == "real_time":
            from runner.tacview import Tacview
            self.tacview = Tacview()

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.num_env_steps = int(self.all_args.num_env_steps)
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.buffer_size = self.all_args.buffer_size
        self.use_wandb = self.all_args.use_wandb

        # interval
        self.save_interval = self.all_args.save_interval
        self.log_interval = self.all_args.log_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.eval_episodes = self.all_args.eval_episodes

        # dir
        self.model_dir = self.all_args.model_dir
        self.run_dir = config["run_dir"]
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.save_dir = str(self.run_dir)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.load()
        
    def load(self):
        # algorithm
        if self.algorithm_name == "ppo":
            from ..algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from ..algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args,
                             self.envs.observation_space,
                             self.envs.action_space,
                             device=self.device)
        self.trainer = Trainer(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args,
                                   self.envs.observation_space,
                                   self.envs.action_space)

        if self.model_dir is not None:
            self.restore()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def rollout(self):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.policy.prep_rollout()
        next_values = self.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                             np.concatenate(self.buffer.rnn_states_critic[-1]),
                                             np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.buffer.n_rollout_threads))
        self.buffer.compute_returns(next_values)

    def train(self):
        self.policy.prep_training()
        train_infos = self.trainer.train(self.policy, self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        policy_actor = self.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_latest.pt")
        policy_critic = self.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_latest.pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_info(self, infos, total_num_steps):
        if self.use_wandb:
            for k, v in infos.items():
                wandb.log({k: v}, step=total_num_steps)
        else:
            pass
        
    def _init_eval_render_config(self):

        """初始化评估渲染配置，返回标准化字典
        
        Returns:
            dict: 渲染配置字典，包含两个可能的结构：
                - histroy_acmi模式: {'mode': 'txt', 'filepath': '.../file.acmi'}
                - real_time模式: {'mode': 'real_time', 'tacview': tacview_obj}
                或 None（当不需要渲染时）
        """
        if not self.eval_render_mode:
            return None

        config = {}
        
        if self.eval_render_mode == "histroy_acmi":
            config.update({
                'mode': 'txt',
                'filepath': os.path.join(
                    self.run_dir,
                    'acmi_files',
                    f'eval_episode_{self.current_episode}.acmi'
                )
            })
            os.makedirs(os.path.dirname(config['filepath']), exist_ok=True)
        
        elif self.eval_render_mode == "real_time":
            if not self.tacview:
                return None
            print("Reconnecting TacView...")
            self.tacview.reconnect()
            config.update({
                'mode': 'real_time',
                'tacview': self.tacview
            })
        
        else:
            raise ValueError(f"不支持的渲染模式: {self.eval_render_mode}")

        return config
    
    def eval_render(self):
        # eval render with tacview
        if self.eval_render_config:
            _config = self.eval_render_config 
            if _config['mode'] == 'txt' and self.current_episode % self.eval_interval == 0:
                self.eval_envs.envs[0].render(_config['mode'], _config['filepath'])
            elif _config['mode'] == 'real_time':
                self.eval_envs.envs[0].render(_config['mode'], "None", _config['tacview'])