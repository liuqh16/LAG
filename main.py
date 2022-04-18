import gym
import torch
import numpy as np
import os
import sys
import logging
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from ppo_trainer import PPOTrainer
from ppo_policy import PPOPolicy
from ppo_data_collectors import BaseDataCollector, SelfPlayDataCollector
from config import get_config
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv

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

def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    all_args.buffer_size = 3000
    all_args.env_name = 'SingleCombat'
    all_args.eval_episodes = 1
    all_args.num_env_steps = 1e8
    all_args.num_mini_batch = 5
    all_args.ppo_epoch = 4
    all_args.cuda = True
    all_args.lr = 1e-4

    env = make_env(all_args)
    collector = SelfPlayDataCollector(all_args)
    trainer = PPOTrainer(all_args, env.observation_space, env.action_space)
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / all_args.env_name /all_args.scenario_name / all_args.experiment_name / time.strftime("%b%d-%H%M%S", time.localtime())
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    # writer = SummaryWriter(run_dir)
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(all_args).items()])),
    # )
    torch.save(trainer.policy.params(), f"{run_dir}/agent_0.pt")
    num_epochs = int(all_args.num_env_steps // all_args.buffer_size)
    for epoch in range(num_epochs):
        # train
        params = torch.load(f"{str(run_dir)}/agent_{epoch}.pt")
        _, buffer = collector.collect_data(ego_params=params, enm_params=params)

        _, (params, train_info) = trainer.train(params=params, buffer=buffer)

        # eval and record info
        _, (elo_gain, eval_info) = collector.evaluate_data(ego_params=params, enm_params=params)

        cur_steps = (epoch + 1) * all_args.buffer_size
        info = {**train_info, **eval_info}
        # for k, v in info.items():
        #     writer.add_scalar(f"info/{k}", v, global_step=cur_steps)
        train_reward, eval_reward = train_info['episode_reward'], eval_info['episode_reward']
        print(f"Epoch {epoch} / {num_epochs} , train episode reward {train_reward}, evaluation episode reward {eval_reward}")

        # save
        torch.save(params, f"{str(run_dir)}/agent_{epoch+1}.pt")

if __name__ == '__main__':
    main(sys.argv[1:])




