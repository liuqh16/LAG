import math
import numpy as np
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        self.target_angles = np.array([30., 60., 90., 120., 150., 180.])
        self.check_interval = 20

    def get_termination(self, task, env, agent_id, info={}):
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
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10:
                done = True
            # if current target heading is reached, random generate a new target heading
            else:
                index = env.np_random.randint(0, 6)
                angle = self.target_angles[index]
                sign = env.np_random.choice([-1, +1])
                new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + sign * angle
                new_heading = (new_heading + 360) % 360
                env.heading_turns += 1
                self.log(f'current_step:{cur_step} target_heading:{new_heading}')
                env.agents[agent_id].set_property_value(c.target_heading_deg, new_heading)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
        if done:
            self.log(f'INFO: agent[{agent_id}] unreached heading, Total Steps={env.current_step}')
            info['heading_turns'] = env.heading_turns
            info[f'agent{agent_id}_end_reason'] = 3  # unreach_heading
        success = False
        return done, success, info
