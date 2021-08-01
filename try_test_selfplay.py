import torch
import os
import argparse
import signal
import random
import numpy as np

from algorithms.ppo_data_collectors import SelfPlayDataCollector
from algorithms.ppo_args_baseline import Config
from envs.JSBSim.envs.selfplay_env import SelfPlayEnv
from envs.JSBSim.core.render_tacview import data_replay

def main():
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="JSBSim")
    parser.add_argument("--task", default="selfplay")
    parser.add_argument("--version", default='v0')
    parser.add_argument("--iter", default=0, type=int)  # select model
    parser.add_argument("--gpu-id", default=None, type=int)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    filepath = f'./results/{args.env}_{args.task}_{args.version}/models/agent{0}_history{args.iter}.pt'
    assert os.path.exists(filepath), 'ModelFile does not exist!'

    args_ppo = Config(env=SelfPlayEnv())
    if args.gpu_id is None:
        args_ppo.device = torch.device('cpu')
    else:
        args_ppo.device = torch.device(f'cuda:{args.gpu_id}')
    hyper_params = dict()
    hyper_params['reward_hyper'] = [1]
    hyper_params['ppo_hyper'] = [1., 1.]
    collector = SelfPlayDataCollector(args_ppo)
    trajectory_data = collector.collect_data_for_show(ego_net_params=torch.load(filepath)['model_state_dict'],
                                                      enm_net_params=torch.load(filepath)['model_state_dict'])
    # np.save(f'./results/{args.env}_{args.task}_{args.version}/render_data', trajectory_data)
    data_replay(trajectory_data)

if __name__ == '__main__':
    main()
