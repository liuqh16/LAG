import numpy as np
from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):
    """
    Base RewardFunction class
    Reward-specific reset and get_reward methods are implemented in subclasses
    """
    def __init__(self, config, is_potential=False, render=False):
        self.config = config
        self.reward_scale = 1.0
        self.is_potential = is_potential
        self.render = render
        # inner variables
        self.pre_reward = 0.0
        self.reward_trajectory = []  # type: list[tuple[float]]
        self.reward_item_names = [self.__class__.__name__]

    def reset(self, task, env, agent_id=0):
        """Perform reward function-specific reset after episode reset.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance
        """
        if self.is_potential:
            self.pre_reward = self.get_reward(task, env, agent_id)
        if self.render:
            self.reward_trajectory = []

    @abstractmethod
    def get_reward(self, task, env, agent_id=0):
        """Compute the reward at the current timestep.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        raise NotImplementedError

    def _process(self, new_reward, render_items=None):
        if self.is_potential:
            reward_diff = new_reward - self.pre_reward
            self.pre_reward = new_reward
            reward = reward_diff * self.reward_scale
        else:
            reward = new_reward * self.reward_scale
        if self.render:
            if render_items is None:
                self.reward_trajectory.append(reward)
            else:
                self.reward_trajectory.append((reward, *render_items))
        return reward

    def get_reward_trajectory(self):
        """Get all the reward history of current episode.py

        Returns:
            (dict): {reward_name(str): reward_trajectory(np.array)}
        """
        assert self.render, "Must set self.render=True to track!"
        if len(self.reward_item_names) == 1:
            return {self.reward_item_names[0]: np.array(self.reward_trajectory)}
        else:
            return dict(zip(self.reward_item_names, np.array(list(zip(*self.reward_trajectory)))))
