from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c
import math
import random


class UnreachHeadingAndAltitude(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        self.check_interval = self.config.init_config[0]['heading_check_interval']

    def get_termination(self, task, env, agent_id=0, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        done = False
        success = False
        ego_uid = list(env.jsbsims.keys())[agent_id]
        check_time = env.jsbsims[ego_uid].get_property_value(c.heading_check_time)
        if env.jsbsims[ego_uid].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.jsbsims[ego_uid].get_property_value(c.delta_altitude)) >= 300:
                done = True
                print(info)
                print(f'INFO: agent[{agent_id}] unreached attitude!')
                info[f'agent{agent_id}_end_reason'] = 1

            if math.fabs(env.jsbsims[ego_uid].get_property_value(c.delta_heading)) > 10:
                done = True
                print(f'INFO: agent[{agent_id}] unreached heading!')
                print(info)
                info[f'agent{agent_id}_end_reason'] = 1
            
            # Change target angle every check_interval seconds
            angle = random.choice([30., 60., 90., 120., 150., 180.])
            sign = random.choice([+1.0, -1.0])
            new_heading = env.jsbsims[ego_uid].get_property_value(c.target_heading_deg) + sign * angle
            new_heading = (new_heading + 360) % 360
            env.jsbsims[ego_uid].set_property_value(c.target_heading_deg, new_heading)

            # Change target altitude every check_interval seconds
            alt = random.choice([1000, 2000, 3000, 4000])
            sign = random.choice([+1.0, -1.0])
            new_alt = env.jsbsims[ego_uid].get_property_value(c.target_altitude_ft) + sign * alt
            new_alt = max(new_alt, 15000)
            env.jsbsims[ego_uid].set_property_value(c.target_altitude_ft, new_alt)
            env.jsbsims[ego_uid].set_property_value(c.heading_check_time, check_time + self.check_interval)

            info[f'time{check_time}_target_heading'] = new_heading
            info[f'time{check_time}_target_altitude'] = new_alt

        success = False
        return done, success, info