import os
import random
import shutil
import numpy as np
from copy import deepcopy


def population_based_exploit(epoch: int, elos: dict, hypers: dict, run_dir: str):
    topk = int(np.ceil(0.2 * len(elos)))
    elo_threshold = 50
    elos_new = deepcopy(elos)
    hypers_new = deepcopy(hypers)
    ranks = {a: elos[a][epoch+1] for a in elos.keys()}
    sorted_ranks_idxs = [k for k in sorted(ranks, key=ranks.__getitem__, reverse=True)]
    topk_idxs = list(sorted_ranks_idxs[:topk])
    for agent_id in elos_new.keys():
        agent_elo = elos[agent_id][epoch+1]
        better_agent_id = random.sample(topk_idxs, 1)[0]
        if len(sorted_ranks_idxs) == 1:
            # population size = 1, no exploit
            break
        if ranks[better_agent_id] - agent_elo < elo_threshold or agent_id in topk_idxs:
            # the agent is already good enough
            continue
        elos_new[agent_id][epoch+1] = elos[better_agent_id][epoch+1]
        os.remove(f"{run_dir}/agent{agent_id}_history{epoch+1}.pt")
        os.remove(f"{run_dir}/agent{agent_id}_latest.pt")
        shutil.copy(f"{run_dir}/agent{better_agent_id}_latest.pt", f"{run_dir}/agent{agent_id}_history{epoch+1}.pt")
        shutil.copy(f"{run_dir}/agent{better_agent_id}_latest.pt", f"{run_dir}/agent{agent_id}_latest.pt")
        for s in hypers[agent_id].keys():
            hyper_ = hypers[agent_id][s]
            new_hyper_ = hypers[better_agent_id][s]
            for i in range(len(hyper_)):
                inherit_prob = np.random.binomial(1, 0.5, 1)[0]
                hyper_tmp = (1. - inherit_prob) * hyper_[i] + inherit_prob * new_hyper_[i]
                hypers_new[agent_id][s][i] = float(hyper_tmp)
    return elos_new, hypers_new


def population_based_explore():
    pass