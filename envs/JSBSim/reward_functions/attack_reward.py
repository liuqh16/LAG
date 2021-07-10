import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class AttackReward(BaseRewardFunction):
    """
    AttackReward
    Use airborne weapon(aircraft artillery) to attack the enemy fighter, gain reward if enemy's blood decreases.

    NOTE:
    - Only support one-to-one environments.
    - env must implement `self.features` property
    """
    def __init__(self, config, is_potential=False, render=False):
        super().__init__(config, is_potential, render)
        assert len(self.config.init_config.keys()) == 2, \
            "OrientationReward only support one-to-one environments but current env has more than 2 agents!"
        self.reward_scale = getattr(self.config, 'attack_reward_scale', 1.0)
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
        ego_name, enm_name = env.agent_names[agent_id], env.agent_names[(agent_id + 1) % env.num_agents]
        ego_feature, enm_feature = env.features[ego_name], env.features[enm_name]
        enm_AO, enm_TA, R = get_AO_TA_R(enm_feature, ego_feature)
        delta_blood = 0
        if np.abs(np.rad2deg(enm_TA)) < 60 and np.abs(np.rad2deg(enm_AO)) < 30 and R <= 3:
            delta_blood = -10 * 0
        task.bloods[ego_name] += delta_blood
        new_reward = -delta_blood
        return self._process(new_reward)
