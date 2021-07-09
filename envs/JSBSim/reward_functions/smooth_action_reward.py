import numpy as np
from .reward_function_base import BaseRewardFunction


class SmoothActionReward(BaseRewardFunction):
    """
    SmoothActionReward
    Punish if current fighter change action significantly. Typically negative.

    NOTE:
    - env must implement `self.features` property
    """
    def __init__(self, config, is_potential=False, render=False):
        super().__init__(config, is_potential, render)
        self.reward_scale = getattr(self.config, 'smooth_action_reward_scale', 1.0)
        self.pre_actions = None

        self.reward_item_names = [self.__class__.__name__]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        if self.pre_actions is None:
            self.pre_actions = env.actions
        ego_name = env.agent_names[agent_id]
        cur_action, pre_action = env.actions[ego_name], self.pre_actions[ego_name]
        delta_action = np.abs(cur_action - pre_action)
        new_reward = -np.mean(delta_action * (delta_action > 10)) * 0.001
        return self._process(new_reward)
