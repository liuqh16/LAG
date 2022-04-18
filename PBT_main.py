import numpy as np
import torch
import ray
import time
import os
import sys
import gym
import random
import logging
import setproctitle

from config import get_config
import envs
from pathlib import Path
from pipeline_actor_pool import PipelineActorPool
from ppo_data_collectors import DataCollectorMix
from ppo_trainer import PBTPPOTrainer
from ppo_policy import PPOPolicy
from util_population import population_based_exploit
from util_selfplay import choose_opponents
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv

def load_enm_params(run_dir: str, enm_idx: tuple):
    agent_id, t = enm_idx
    return torch.load(f"{run_dir}/agent{agent_id}_history{t}.pt", map_location=torch.device('cpu'))

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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed(all_args.seed)

    all_args.buffer_size = 3000
    all_args.cuda = True
    all_args.population_size = 1
    all_args.num_parallel_each_agent = 16
    num_env_parallels = all_args.population_size * all_args.num_parallel_each_agent
    all_args.init_elo = 1000

    setproctitle.setproctitle(str(all_args.env_name)+'@'+ str(all_args.user_name))
    env = make_env(all_args)

    ray.init()
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "runs" / all_args.env_name / all_args.experiment_name / time.strftime("%b%d-%H%M%S", time.localtime())
    if not run_dir.exists():
        os.makedirs(run_dir)
    run_dir = str(run_dir)

    data_collectors = [DataCollectorMix.remote(all_args) for _ in range(num_env_parallels)]
    data_collector_pools = PipelineActorPool(data_collectors, [])
    ppo_trainers = [PBTPPOTrainer.remote(all_args, env.observation_space, env.action_space)]
    ppo_trainer_pools = PipelineActorPool(ppo_trainers, [data_collector_pools])

    population = {}
    population_elos = {}
    population_hypers = {}
    for agent_id in range(all_args.population_size):
        population[agent_id] = PPOPolicy(all_args, env.observation_space, env.action_space)
        params = population[agent_id].params()
        torch.save(params, f'{run_dir}/agent{agent_id}_history0.pt')
        torch.save(params, f'{run_dir}/agent{agent_id}_latest.pt')
        population_elos[agent_id] = {0: all_args.init_elo}
        population_hypers[agent_id] = dict() 
    logging.info('init over')

    num_epochs = int(all_args.num_env_steps // all_args.buffer_size // all_args.num_parallel_each_agent)
    for epoch in range(num_epochs):
        # 1. data collector for training
        vs = []
        for agent_id in range(all_args.population_size):
            enm_idxs, enm_elos = choose_opponents(agent_id, population_elos, all_args.num_parallel_each_agent)
            # vs.extend([dict(ego_params=torch.load(f"{str(run_dir)}/model_{epoch}.pt"))] * num_env_parallels)
            vs.extend([dict(ego_params=torch.load(f"{run_dir}/agent{agent_id}_latest.pt", map_location=torch.device('cpu')),
                            enm_params=load_enm_params(run_dir, enm_idxs[i]),
                            hyper_params=population_hypers[agent_id]) 
                            for i in range(len(enm_idxs))])
        data_collector_pools.timeout = 100
        data_collector_pools.map(lambda a, _, v: a.collect_data.remote(**v),
                                    [[] for _ in range(len(vs))],
                                    vs,
                                    [(i, j) for i in range(all_args.population_size) for j in range(all_args.num_parallel_each_agent)])
        
        # 2. ppo train
        vs = []
        for agent_id in range(all_args.population_size):
            vs.append(dict(params=torch.load(f"{str(run_dir)}/agent{agent_id}_latest.pt", map_location=torch.device('cpu')),
                           hyper_params=population_hypers[agent_id]))
        ppo_trainer_pools.map(lambda a, data, v: a.train.remote(buffer=data, **v),
                                [[(0, (i, j)) for j in range(all_args.num_parallel_each_agent)] for i in range(all_args.population_size)],
                                vs,
                                list(range(all_args.population_size)))
        for agent_id, result in ppo_trainer_pools.wait_generator():
            if result is not None:
                params, train_info = result
            else:
                NotImplementedError("Train Object loss")
            train_reward = train_info["episode_reward"]
            logging.info(f"Epoch {epoch} / {num_epochs}, Agent{agent_id}, train episode reward {train_reward}")
            torch.save(params, f'{run_dir}/agent{agent_id}_history{epoch+1}.pt')
            torch.save(params, f'{run_dir}/agent{agent_id}_latest.pt')
        data_collector_pools.flush()

        # 3. data collector for evaluating
        vs = []
        enm_infos = {} 
        eval_gains = [None for k in range(all_args.population_size * all_args.num_parallel_each_agent)]
        eval_rewards = [None for k in range(all_args.population_size * all_args.num_parallel_each_agent)]
        for agent_id in range(all_args.population_size):
            enm_idxs, enm_elos = choose_opponents(agent_id, population_elos, all_args.num_parallel_each_agent)
            enm_infos[agent_id] = enm_idxs
            vs.extend([dict(ego_params=torch.load(f"{run_dir}/agent{agent_id}_latest.pt", map_location=torch.device('cpu')),
                            enm_params=load_enm_params(run_dir, enm_idxs[i]),
                            hyper_params=population_hypers[agent_id],
                            ego_elo=population_elos[agent_id][epoch],
                            enm_elo=enm_elos[i]
                            ) for i in range(len(enm_idxs))])
        data_collector_pools.timeout = 100
        data_collector_pools.map(lambda a, _, v: a.evaluate_data.remote(**v),
                                [[] for _ in range(len(vs))],
                                vs,
                                list(range(all_args.population_size * all_args.num_parallel_each_agent)))
        for i, result in data_collector_pools.wait_generator():
            agent_id = i // all_args.num_parallel_each_agent
            elo_gain, eval_info = result
            reward = eval_info['episode_reward']
            eval_gains[i] = elo_gain
            eval_rewards[i] = reward

        for agent_id in range(all_args.population_size):
            avg_eval_reward = 0.
            population_elos[agent_id][epoch+1] = population_elos[agent_id][epoch]
            for i, enm_idxs in enumerate(enm_infos[agent_id]):
                elo_gain = eval_gains[agent_id * all_args.num_parallel_each_agent + i]
                avg_eval_reward += eval_rewards[agent_id * all_args.num_parallel_each_agent + i]
                population_elos[agent_id][epoch+1] += elo_gain
                enm_id, enm_t = enm_idxs
                population_elos[enm_id][enm_t] -= elo_gain
            avg_eval_reward /= len(enm_infos[agent_id])
            logging.info(f"Epoch {epoch} / {num_epochs}, Agent{agent_id}, eval episode reward {avg_eval_reward}")

        # 4. population based exploit
        population_elos, population_hypers = population_based_exploit(epoch, population_elos, population_hypers, run_dir)
        # TODO: 5. population based explore 
        # 6. save checkoutpoint
        checkpoint = {
            "epoch": epoch,
            "population_elos": population_elos,
            "population_hypers": population_hypers
        }
        torch.save(checkpoint, f"{run_dir}/checkpoint_latest.pt")
        env.close()

        
if __name__ == "__main__":
    main(sys.argv[1:])

    

    
