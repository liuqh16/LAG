import numpy as np
from typing import Dict, List
from abc import ABC, abstractstaticmethod


def get_algorithm(algo_name):
    if algo_name == 'sp':
        return SP
    elif algo_name == 'fsp':
        return FSP
    elif algo_name == 'pfsp':
        return PFSP
    else:
        raise NotImplementedError("Unknown algorithm {}".format(algo_name))


class SelfplayAlgorithm(ABC):

    @abstractstaticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        pass

    @abstractstaticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class SP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return list(agents_elo.keys())[-1]

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class FSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], **kwargs) -> str:
        return np.random.choice(list(agents_elo.keys()))

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]], **kwargs) -> None:
        pass


class PFSP(SelfplayAlgorithm):

    @staticmethod
    def choose(agents_elo: Dict[str, float], lam=1, s=100, **kwargs) -> str:
        history_elo = np.array(list(agents_elo.values()))
        sample_probs = 1. / (1. + 10. ** (-(history_elo - np.median(history_elo)) / 400.)) * s
        """ meta-solver """
        k = float(len(sample_probs) + 1)
        meta_solver_probs = np.exp(lam / k * sample_probs) / np.sum(np.exp(lam / k * sample_probs))
        opponent_idx = np.random.choice(a=list(agents_elo.keys()), size=1, p=meta_solver_probs).item()
        return opponent_idx

    @staticmethod
    def update(agents_elo: Dict[str, float], eval_results: Dict[str, List[float]]) -> None:
        pass
