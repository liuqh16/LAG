from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c
import math
import random


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        self.agent_names = list(self.config.init_config.keys())

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
        ego_name = self.agent_names[agent_id]
        done = False
        success = False

        temp = env.sims[ego_name].get_property_value(c.steady_flight)
        if env.sims[ego_name].get_property_value(c.simulation_sim_time_sec) >= temp:
            if math.fabs(env.sims[ego_name].get_property_value(c.delta_heading)) > 10:
                done = True
            if math.fabs(env.sims[ego_name].get_property_value(c.delta_altitude)) >= 100:
                done = True
            # Change heading every 150 seconds
            angle = int(env.sims[ego_name].get_property_value(c.steady_flight) / 150) * 10
            sign = random.choice([+1.0, -1.0])
            new_heading = env.sims[ego_name].get_property_value(c.target_heading_deg) + sign * angle
            new_heading = (new_heading + 360) % 360

            # print(f'Time to change: {sim.get_property_value(c.simulation_sim_time_sec)} (Heading: {sim.get_property_value(c.target_heading_deg)} -> {new_heading})')

            env.sims[ego_name].set_property_value(c.target_heading_deg, new_heading)

            env.sims[ego_name].set_property_value(c.steady_flight, env.sims[ego_name].get_property_value(c.steady_flight) + 150)

        if done:
            print(f'INFO: [{env.agent_names[agent_id]}] unreached heading!')
            info[f'{env.agent_names[agent_id]}_end_reason'] = 1  # crash
        success = False
        return done, success, info
