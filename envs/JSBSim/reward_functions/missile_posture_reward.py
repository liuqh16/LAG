import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.v = 0 # record the latest missile velocity, avoid reward mutation when missile is deleted

    def get_reward(self, task, env, agent_id):
        """
        Reward is velocity attenuation of the missile

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        missile_sim = task.check_missile_warning(env, agent_id)
        if missile_sim is not None:
            self.v = (np.linalg.norm(missile_sim.get_velocity()) / 340) * 100
        reward = self.v
        return self._process(reward, agent_id)


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
        reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        reward = max(-reward, 0)
        self.reward_trajectory[agent_id].append([reward, *render_items])
        return reward
