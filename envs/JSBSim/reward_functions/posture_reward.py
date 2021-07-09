import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    - env must implement `self.features` property
    """
    def __init__(self, config, is_potential=False, render=False):
        super().__init__(config, is_potential, render)
        assert len(self.config.init_config.keys()) == 2, \
            "OrientationReward only support one-to-one environments but current env has more than 2 agents!"
        self.reward_scale = getattr(self.config, 'posture_reward_scale', 1.0)
        self.orientation_version = getattr(self.config, 'orientation_version', 'v2')
        self.range_version = getattr(self.config, 'range_version', 'v1')
        self.target_dist = getattr(self.config, 'target_dist', 3.0)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_name, enm_name = env.agent_names[agent_id], env.agent_names[(agent_id + 1) % env.num_agents]
        ego_feature, enm_feature = env.features[ego_name], env.features[enm_name]
        AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
        orientation_reward = self.orientation_fn(AO, TA)
        range_reward = self.range_fn(R)
        new_reward = orientation_reward * range_reward
        return self._process(new_reward, agent_id=agent_id, render_items=(orientation_reward, range_reward))

    def get_orientation_function(self, version):
        if version == 'v0':
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) / (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
