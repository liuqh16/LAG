import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward = v_reward + d_reward:
    v_reward: [-1,0], the bigger missile velocity, the less reward
    d_reward: [-1,0], the longer distance between flighter and missile, the more reward

    NOTE:
    - Only support one-to-one environments.
    - MissileAttackReward will block other reward functions, so must be placed on top!
    """
    def __init__(self, config):
        super().__init__(config)
        # self.all_reward_scales = []
        self.v_fn = lambda v: np.exp(-v / 200) - 1
        self.d_fn = lambda d: -np.exp(-d / 1000)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_v', '_d']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward, v_reward, d_reward = 0, 0, 0
        missile_sim = task.check_missile_warning(env, agent_id)
        if missile_sim is not None:
            v_reward = self.v_fn(np.linalg.norm(missile_sim.get_velocity()))
            d_reward = self.d_fn(missile_sim.target_distance)
            reward = v_reward + d_reward
        return self._process(reward, agent_id, (v_reward, d_reward))
