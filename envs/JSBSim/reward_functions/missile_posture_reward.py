import numpy as np
from numpy.core.numeric import array_equal
from numpy.lib.arraypad import pad
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R
from ..core.catalog import Catalog


class MissilePostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage not to be pointed at by missile.
    - Range: TODO: Encourage to far away from missile

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        assert self.num_fighters == 2, \
            "PostureReward only support one-to-one environments but current env has more than 2 agents!"
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)

        self.orientation_fn = self.get_orientation_function(self.orientation_version)
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
        ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_fighters
        if not task.missile_lists[enm_idx].missile_info[0]['flying']:
            return 0
        else:
            # feature: (north, east, down, vn, ve, vd) unit: km, mh
            ego_feature = np.hstack([env.sims[ego_idx].get_position() / 1000, env.sims[ego_idx].get_velocity() / 340])
            enm_missile_feature = task.missile_lists[enm_idx].missile_info[0]['current_state'] # get missile state TODO: if there are multi missiles
            enm_missile_feature = task.missile_lists[enm_idx].simulator.transfer2raw(enm_missile_feature) # transform coordinate system
            enm_missile_feature = np.array([*(enm_missile_feature[:3]/1000), *(enm_missile_feature[3:6]/340)]) # transform unit

            AO, TA, R = get_AO_TA_R(ego_feature, enm_missile_feature)
            orientation_reward = self.orientation_fn(AO, TA)
            new_reward = orientation_reward
            return self._process(new_reward, agent_id)

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
