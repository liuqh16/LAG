import torch
import os
import argparse
import signal
import random
import numpy as np

from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.ppo_data_collectors import SelfPlayDataCollector
from algorithms.ppo_training_agent import Trainer
from algorithms.ppo_AC import ActorCritic
from algorithms.ppo_args import Config


def make_train_env(num_env, taskname):
    def env_fn():
        return SingleCombatEnv(config=taskname)
    return SubprocVecEnv([env_fn for _ in range(num_env)])


def main():
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)
    parser = argparse.ArgumentParser()
    # env config
    parser.add_argument("--env", default="JSBSim")
    parser.add_argument("--task", default="selfplay")
    parser.add_argument("--exp", default='default')
    # train config
    parser.add_argument("--num-parallel-env", default=5, type=int)
    parser.add_argument("--num-train", default=1000, type=int)
    parser.add_argument("--num-eval", default=10, type=int)
    parser.add_argument("--agent-id", default=0, type=int)
    parser.add_argument("--gpu-id", default=None, type=int)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = f'./results/{args.env}_{args.task}/{args.exp}'
    if not os.path.exists(rootpath):
        os.makedirs(f'{rootpath}/models')

    envs = make_train_env(args.num_parallel_env, f'{args.task}_task')
    args_ppo = Config(env=envs)
    if args.gpu_id is None:
        args_ppo.device = torch.device('cpu')
    else:
        args_ppo.device = torch.device(f'cuda:{args.gpu_id}')
    hyper_params = dict()
    hyper_params['ppo_hyper'] = args_ppo.ppo_hyper

    collector = SelfPlayDataCollector(args_ppo)
    trainer = Trainer(args_ppo)
    agent = ActorCritic(args_ppo, agent_idx=args.agent_id)
    agent.save_model(rootpath, None, epoch_t=0, args=args)
    rewards_list = []
    for itr in range(args.num_train):
        res = collector.collect_data(ego_net_params=torch.load(f"{rootpath}/models/agent{args.agent_id}_latest.pt")['model_state_dict'],
                                     enm_net_params=torch.load(f"{rootpath}/models/agent{args.agent_id}_latest.pt")['model_state_dict'],
                                     hyper_params=hyper_params, agent_id=args.agent_id)
        status, sample_data = res
        params = trainer.update_agent(agent_id=args.agent_id,
                                      agent_params=torch.load(f"{rootpath}/models/agent{args.agent_id}_latest.pt"),
                                      buffer_data_lists=sample_data, hyper_params=hyper_params)
        torch.save(params, f"{rootpath}/models/agent{args.agent_id}_latest.pt")
        torch.save(params, f"{rootpath}/models/agent{args.agent_id}_history{itr + 1}.pt")
        rewards = collector.evaluate_with_baseline(ego_net_params=torch.load(f"{rootpath}/models/agent{args.agent_id}_latest.pt"),
                                                   enm_net_params=torch.load(f"{rootpath}/models/agent{args.agent_id}_latest.pt"),
                                                   eval_num=args.num_eval)
        rewards_list.append(rewards)
        np.save(f'{rootpath}/learning_reward', np.asarray(rewards_list))


if __name__ == '__main__':
    main()
