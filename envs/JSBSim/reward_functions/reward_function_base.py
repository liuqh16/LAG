import numpy as np
from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    def __init__(self, config):
        self.config = config
        # inner variables
        self.num_aircrafts = len(getattr(self.config, 'aircraft_configs', {}).keys())
        self.reward_scale = getattr(self.config, f'{self.__class__.__name__}_scale', 1.0)
        self.is_potential = getattr(self.config, f'{self.__class__.__name__}_potential', False)
        self.pre_rewards = [0.0 for _ in range(self.num_aircrafts)]
        self.reward_trajectory = [[] for _ in range(self.num_aircrafts)]
        self.reward_item_names = [self.__class__.__name__]

    def reset(self, task, env):
        """Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        if self.is_potential:
            self.pre_rewards = [0.0 for _ in range(self.num_aircrafts)]
            for agent_id in range(self.num_aircrafts):
                self.pre_rewards[agent_id] = self.get_reward(task, env, agent_id)
        self.reward_trajectory = [[] for _ in range(self.num_aircrafts)]

    @abstractmethod
    def get_reward(self, task, env, agent_id):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        raise NotImplementedError

    def _process(self, new_reward, agent_id=0, render_items=()):
        """Process reward and inner variables.

        Args:
            new_reward (float)
            agent_id (int)
            render_items (tuple, optional): Must set if `len(reward_item_names)>1`. Defaults to None.

        Returns:
            [type]: [description]
        """
        reward = new_reward * self.reward_scale
        if self.is_potential:
            reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        self.reward_trajectory[agent_id].append([reward, *render_items])
        return reward

    def get_reward_trajectory(self):
        """Get all the reward history of current episode.py

        Returns:
            (dict): {reward_name(str): reward_trajectory(np.array)}
        """
        return dict(zip(self.reward_item_names, np.array(self.reward_trajectory).transpose(2, 0, 1)))
