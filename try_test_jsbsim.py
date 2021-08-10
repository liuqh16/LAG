import torch
import os
import argparse
import signal
import random
import numpy as np

from algorithms.ppo_data_collectors import SelfPlayDataCollector
from algorithms.ppo_args import Config
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.core.render_tacview import data_replay


def make_test_env(taskname):
    return SingleCombatEnv(config=taskname)


def main():
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="JSBSim")
    parser.add_argument("--task", default="singlecombat")
    parser.add_argument("--exp", default='test')
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--modelpath", default=None, type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    filepath = args.modelpath
    assert filepath is not None, "Must select a model to load!"
    assert os.path.exists(filepath), 'ModelFile does not exist!'

    envs = make_test_env(f'{args.task}_task')
    args_ppo = Config(env=envs)
    args_ppo.device = torch.device('cpu')
    hyper_params = dict()
    hyper_params['reward_hyper'] = [1]
    hyper_params['ppo_hyper'] = [1., 1.]
    collector = SelfPlayDataCollector(args_ppo)
    trajectory_data = collector.collect_data_once(ego_net_params=torch.load(filepath, map_location='cpu')['model_state_dict'],
                                                  enm_net_params=torch.load(filepath, map_location='cpu')['model_state_dict'])
    if not os.path.exists(f'./results/{args.env}_{args.task}/{args.exp}'):
        os.makedirs(f'./results/{args.env}_{args.task}/{args.exp}')
    np.save(f'./results/{args.env}_{args.task}/{args.exp}/render_data', trajectory_data)
    # data_replay(trajectory_data)

if __name__ == '__main__':
    main()
