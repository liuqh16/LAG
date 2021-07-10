import numpy as np
from collections import OrderedDict
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
from ..utils.missile_utils import Missile3D


class MissileAttackReward(BaseRewardFunction):
    """
    MissileAttackReward
    Use airborne missile to attack the enemy fighter, gain reward if enemy's blood decreases.

    NOTE:
    - Only support one-to-one environments.
    - env must implement `self.features` property
    - MissileAttackReward will block other reward functions, so must be placed on top!
    """
    def __init__(self, config):
        super().__init__(config)
        assert len(self.config.init_config.keys()) == 2, \
            "MissileAttackReward only support one-to-one environments but current env has more than 2 agents!"
        self.missile_models = OrderedDict([(agent, Missile3D()) for agent in self.agent_names])
        self.all_reward_scales = []

    def reset(self, task, env):
        super().reset(task, env)
        self.missile_models = OrderedDict([(agent, Missile3D()) for agent in self.agent_names])
        for reward_fn in task.reward_functions:
            self.all_reward_scales.append(reward_fn.reward_scale)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_name, enm_name = self.agent_names[agent_id], self.agent_names[(agent_id + 1) % self.num_agents]
        missile_info, target_info = env.features[ego_name], env.features[enm_name]
        missile_info[0:3] *= 1000   # unit: m
        missile_info[3:6] *= 340    # unit: m/s
        target_info[0:3] *= 1000    # unit: m
        target_info[3:6] *= 340     # unit: m/s
        missile_pos_vel, info = self.missile_models[ego_name].missile_step(missile_info, target_info)
        if info['mask_enm']:
            for reward_fn in task.reward_functions:
                if not isinstance(reward_fn, MissileAttackReward):
                    reward_fn.reward_scale = 0.
            # TODO: how to calculate missile reward?
            new_reward = 0.0
        else:
            for i, reward_fn in enumerate(task.reward_functions):
                reward_fn.reward_scale = self.all_reward_scales[i]
            new_reward = 0.0
        return self._process(new_reward)
