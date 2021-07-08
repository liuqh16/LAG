from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    def __init__(self, config, is_potential=False):
        self.config = config
        self.is_potential = is_potential
        self.pre_reward = 0.0

    def reset(self, task, env):
        """Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        self.pre_reward = 0.0

    @abstractmethod
    def get_reward(self, task, env, agent_id=0):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (reward, info)
        """
        raise NotImplementedError
