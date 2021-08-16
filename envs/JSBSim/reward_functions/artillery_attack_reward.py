import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class ArtilleryAttackReward(BaseRewardFunction):
    """
    ArtilleryAttackReward
    Use airborne artillery to attack the enemy fighter, gain reward if enemy's blood decreases.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        assert self.num_agents == 2, \
            "ArtilleryAttackReward only support one-to-one environments but current env has more than 2 agents!"

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_agents
        # feature: (north, east, down, vn, ve, vd) unit: km, mh
        ego_feature = np.hstack([env.sims[ego_idx].get_position() / 1000, env.sims[ego_idx].get_velocity() / 340])
        enm_feature = np.hstack([env.sims[enm_idx].get_position() / 1000, env.sims[enm_idx].get_velocity() / 340])
        enm_AO, enm_TA, R = get_AO_TA_R(enm_feature, ego_feature)
        delta_blood = 0
        if np.abs(np.rad2deg(enm_TA)) < 60 and np.abs(np.rad2deg(enm_AO)) < 30 and R <= 3:
            delta_blood = -10
        task.bloods[ego_idx] += delta_blood
        new_reward = -delta_blood
        return self._process(new_reward, agent_id)
