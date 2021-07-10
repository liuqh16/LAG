import numpy as np
from .reward_function_base import BaseRewardFunction


class RelativeAltitudeReward(BaseRewardFunction):
    """
    RelativeAltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of relative altitude when larger than 1000  (range: [-1, 0])

    NOTE:
    - Only support one-to-one environments.
    - env must implement `self.features` property
    """
    def __init__(self, config):
        super().__init__(config)
        assert len(self.config.init_config.keys()) == 2, \
            "RelativeAltitudeReward only support one-to-one environments but current env has more than 2 agents!"
        self.KH = getattr(self.config, 'KH', 1.0)     # km

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
        ego_feature, enm_feature = env.features[ego_name], env.features[enm_name]
        ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
        enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
        new_reward = min(self.KH - np.abs(ego_z - enm_z), 0)
        return self._process(new_reward)
