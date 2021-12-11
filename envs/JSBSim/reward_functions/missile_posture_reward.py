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
        self.angle = 0 # angle as well, avoid reward mutation when missile is deleted
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_velocity', '_angle']]

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
            missile_v = missile_sim.get_velocity()
            uid = list(env.jsbsims.keys())[agent_id]
            aircraft_v = env.jsbsims[uid].get_velocity()
            self.angle = np.dot(missile_v, aircraft_v) / (np.linalg.norm(missile_v)*np.linalg.norm(aircraft_v)) 
            self.v = (np.linalg.norm(missile_sim.get_velocity()) / 340) * 100
        reward = self.v * self.reward_scale
        reward, self.pre_rewards[agent_id] = reward - self.pre_rewards[agent_id], reward
        if self.angle < 0:
            reward = self.angle *  1 / (max(-reward, 0) + 1)
        else:
            reward = self.angle * max(-reward, 0)
        return self._process(reward, agent_id, render_items=(self.v, self.angle))


    def _process(self, new_reward, agent_id=0, render_items=()):
        """Process reward and inner variables.

        Args:
            new_reward (float)
            agent_id (int)
            render_items (tuple, optional): Must set if `len(reward_item_names)>1`. Defaults to None.

        Returns:
            [type]: [description]
        """

        self.reward_trajectory[agent_id].append([new_reward, *render_items])
        return new_reward
