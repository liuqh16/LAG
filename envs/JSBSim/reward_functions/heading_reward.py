import numpy as np
import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class HeadingReward(BaseRewardFunction):
    """
    TODO:
    HeadingReward
    Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        """
        TODO:
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        heading_error_scale = 5.0  # degrees
        ego_name = self.agent_names[agent_id]
        heading_r = math.exp(-((env.sims[ego_name].get_property_value(c.delta_heading) / heading_error_scale) ** 2))

        alt_error_scale = 50.0  # feet
        alt_r = math.exp(-((env.sims[ego_name].get_property_value(c.delta_altitude) / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((env.sims[ego_name].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))

        speed_error_scale = 16  # fps (~5%)
        speed_r = math.exp(-(((env.sims[ego_name].get_property_value(c.velocities_u_fps) - 800) / speed_error_scale) ** 2))

        # accel scale in "g"s
        accel_error_scale_x = 0.1
        accel_error_scale_y = 0.1
        accel_error_scale_z = 0.5
        try:
            accel_r = math.exp(
                -(
                    (env.sims[ego_name].get_property_value(c.accelerations_n_pilot_x_norm) / accel_error_scale_x) ** 2
                    + (env.sims[ego_name].get_property_value(c.accelerations_n_pilot_y_norm) / accel_error_scale_y) ** 2
                    + ((env.sims[ego_name].get_property_value(c.accelerations_n_pilot_z_norm) + 1) / accel_error_scale_z) ** 2
                )  # normal value for z component is -1 g
            ) ** (
                1 / 3
            )  # geometric mean
        except OverflowError:
            accel_r = 0

        reward = (heading_r * alt_r * accel_r * roll_r * speed_r) ** (1 / 5)
        return reward