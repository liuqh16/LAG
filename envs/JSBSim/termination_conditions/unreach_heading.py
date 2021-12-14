from numpy.lib.function_base import angle
from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c
import math
import random
import numpy as np


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_heading_increment = aircraft_config['max_heading_increment']
        self.max_altitude_increment = aircraft_config['max_altitude_increment']
        self.max_velocities_u_increment = aircraft_config['max_velocities_u_increment']
        self.check_interval = aircraft_config['check_interval']
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0]

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
        cur_step = info['current_step']
        ego_uid = list(env.jsbsims.keys())[agent_id]
        check_time = env.jsbsims[ego_uid].get_property_value(c.heading_check_time)
        # check heading when simulation_time exceed check_time
        if env.jsbsims[ego_uid].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.jsbsims[ego_uid].get_property_value(c.delta_heading)) > 10:
                done = True
            # if current target heading is reached, random generate a new target heading
            else:
                delta = self.increment_size[env.heading_turn_counts]
                delta_heading = np.random.uniform(-delta, delta) * self.max_heading_increment
                delta_altitude = np.random.uniform(-delta, delta) * self.max_altitude_increment
                delta_velocities_u = np.random.uniform(-delta, delta) * self.max_velocities_u_increment
                new_heading = env.jsbsims[ego_uid].get_property_value(c.target_heading_deg) + delta_heading
                new_heading = (new_heading + 360) % 360
                new_altitude = env.jsbsims[ego_uid].get_property_value(c.target_altitude_ft) + delta_altitude
                new_velocities_u = env.jsbsims[ego_uid].get_property_value(c.target_velocities_u_mps) + delta_velocities_u
                env.jsbsims[ego_uid].set_property_value(c.target_heading_deg, new_heading)
                env.jsbsims[ego_uid].set_property_value(c.target_altitude_ft, new_altitude)
                env.jsbsims[ego_uid].set_property_value(c.target_velocities_u_mps, new_velocities_u)
                env.jsbsims[ego_uid].set_property_value(c.heading_check_time, check_time + self.check_interval)
                env.heading_turn_counts += 1
                print(f'current_step:{cur_step} target_heading:{new_heading}')
                print(f'current_step:{cur_step} target_altitude_ft:{new_altitude}')
                print(f'current_step:{cur_step} target_velocities_u_mps:{new_velocities_u}')
        if done:
            print(f'INFO: agent[{agent_id}] unreached heading, Total Steps={env.current_step}')
            info['heading_turn_counts'] = env.heading_turn_counts
            info[f'agent{agent_id}_end_reason'] = 3  # unreach_heading
        success = False
        return done, success, info
