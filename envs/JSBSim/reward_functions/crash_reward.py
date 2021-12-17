import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c

class CrashReward(BaseRewardFunction):
    """
    CrashReward
    punish if the aircraft is in
    1. extreme state 
    2. low altitude
    3. overload
    """
    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)  # unit: m
        self.acceleration_limit_x = getattr(config, 'acceleration_limit_x', 10.0)  # unit: g
        self.acceleration_limit_y = getattr(config, 'acceleration_limit_y', 10.0)  # unit: g
        self.acceleration_limit_z = getattr(config, 'acceleration_limit_z', 10.0)  # unit: g

    def get_reward(self, task, env, agent_id):
        """
        Reward is -200 if crash else 0

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        ego_uid = list(env.jsbsims.keys())[agent_id]
        is_extreme_state = bool(env.jsbsims[ego_uid].get_property_value(c.detect_extreme_state))
        is_low_altitude  = env.jsbsims[ego_uid].get_property_value(c.position_h_sl_m) <= self.altitude_limit
        is_overload = False
        if env.jsbsims[ego_uid].get_property_value(c.simulation_sim_time_sec) > 10:
            if (math.fabs(env.jsbsims[ego_uid].get_property_value(c.accelerations_n_pilot_x_norm)) > self.acceleration_limit_x
                or math.fabs(env.jsbsims[ego_uid].get_property_value(c.accelerations_n_pilot_y_norm)) > self.acceleration_limit_y
                or math.fabs(env.jsbsims[ego_uid].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > self.acceleration_limit_z
            ):
                is_overload = True
        if is_extreme_state or is_low_altitude or is_overload:
            reward = -200
        return self._process(reward, agent_id)
