from .reward_function_base import BaseRewardFunction


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
        assert self.num_aircrafts == 2, \
            "MissileAttackReward only support one-to-one environments but current env has more than 2 agents!"
        # self.all_reward_scales = []

    def reset(self, task, env):
        super().reset(task, env)
        # for reward_fn in task.reward_functions:
        #     self.all_reward_scales.append(reward_fn.reward_scale)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        # How to calculate missile reward?
        # (1) invoke Missile3D.missile_info to calculate
        # if info['mask_enm']:
        #     for reward_fn in task.reward_functions:
        #         if not isinstance(reward_fn, MissileAttackReward):
        #             reward_fn.reward_scale = 0.
        #     # TODO: how to calculate missile reward?
        #     new_reward = 0.0
        # else:
        #     for i, reward_fn in enumerate(task.reward_functions):
        #         reward_fn.reward_scale = self.all_reward_scales[i]
        #     new_reward = 0.0
        # (2) use task.blood directly
        return self._process(task.bloods[agent_id], agent_id)
