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
    - MissileAttackReward will block other reward functions, so must be placed on top!
    """
    def __init__(self, config):
        super().__init__(config)
        assert self.num_fighters == 2, \
            "MissileAttackReward only support one-to-one environments but current env has more than 2 agents!"
        self.missile_models = [Missile3D() for agent_id in range(self.num_fighters)]
        self.all_reward_scales = []

    def reset(self, task, env):
        super().reset(task, env)
        self.missile_models = [Missile3D() for agent_id in range(self.num_fighters)]
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
        ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_agents
        # feature: (north, east, down, vn, ve, vd) unit: m, m/s
        missile_info = np.hstack([env.sims[ego_idx].get_position(), env.sims[ego_idx].get_velocity()])
        target_info = np.hstack([env.sims[enm_idx].get_position(), env.sims[enm_idx].get_velocity()])
        missile_pos_vel, info = self.missile_models[ego_idx].missile_step(missile_info, target_info)
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
        return self._process(new_reward, agent_id)
