import numpy as np
import random


def _choose_history(history_elo: dict, lam=1., s=100.):
    """
    """
    idx_list = list(history_elo.keys())
    elo_list = np.array(list(history_elo.values()))
    sample_probs = 1. / (1. + 10. ** (-(elo_list - np.median(elo_list)) / 400.)) * s
    """ meta-solver """
    k = float(len(sample_probs) + 1)
    meta_solver_probs = np.exp(lam / k * sample_probs) / np.sum(np.exp(lam / k * sample_probs))

    choose_idx = np.random.choice(a=len(idx_list), size=1, p=meta_solver_probs).item()
    return choose_idx, elo_list[choose_idx]


def choose_opponents(agent_idx: int, agent_elos: dict, num_opponents: int, lam=1., s=100.):
    """
    """
    enm_idxs, enm_history_idxs, enm_elos = [], [], []
    num_total = 1
    while True:
        enm_idx = random.choice([k for k in list(agent_elos.keys())])
        if enm_idx == agent_idx and len(agent_elos) > 1:
            continue
        # 1) choose the opponent agent from populations.
        enm_idxs.append(enm_idx)
        # 2) choose the history copy from the current agent according to ELO
        enm_history_idx, enm_history_elo = _choose_history(agent_elos[enm_idx], lam=lam, s=s)
        enm_history_idxs.append(enm_history_idx)
        enm_elos.append(enm_history_elo)
        num_total += 1
        if num_total > num_opponents:
            break
    enms = []
    for agent, itr in zip(enm_idxs, enm_history_idxs):
            enms.append((agent, itr))
    return enms, enm_elos


if __name__ == '__main__':
    ranks = {
        0: {0: 10, 1: 10, 2: 10, 3: 30, 4: 20, 5: 50, 6: 20},
        1: {0: 80, 1: 60, 2: 30, 3: 90, 4: 20, 5: 50, 6: 20},
        2: {0: 30, 1: 50, 2: 80, 3: 40, 4: 20, 5: 50, 6: 20},
        3: {0: 50, 1: 90, 2: 80, 3: 80, 4: 20, 5: 50, 6: 20},
    }
    ranks1 = {
        0: {0: 10},
        1: {0: 80},
        2: {0: 30},
        3: {0: 50},
    }
    ret = choose_opponents(0, ranks, 4, 5)
    print(ret[0], ret[1])
    print(ret[0][-1])
