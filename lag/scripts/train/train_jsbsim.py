#!/usr/bin/env python
import os
import sys
import random
import socket
import logging
import traceback
from pathlib import Path
import torch
import numpy as np
import wandb
import setproctitle

from lag.config import get_config
from lag.runner.share_jsbsim_runner import ShareJSBSimRunner
from lag.envs.jsbsim import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from lag.envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        if all_args.use_selfplay:
            from lag.runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from lag.runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()

        if all_args.use_wandb:
            run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
