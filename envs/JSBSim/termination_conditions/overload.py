import math
from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class Overload(BaseTerminationCondition):
    """
    Overload
    End up the simulation if acceleration are too high.
    """

    def __init__(self, config):
        super().__init__(config)
        self.acceleration_limit_x = getattr(config, 'acceleration_limit_x', 10.0)  # unit: g
        self.acceleration_limit_y = getattr(config, 'acceleration_limit_y', 10.0)  # unit: g
        self.acceleration_limit_z = getattr(config, 'acceleration_limit_z', 10.0)  # unit: g

    def get_termination(self, task, env, agent_id=0, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if acceleration are too high.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        ego_uid = list(env.jsbsims.keys())[agent_id]
        done = self._judge_overload(env.jsbsims[ego_uid])
        if done:
            print(f'INFO: agent[{agent_id}] acceleration is too high!')
            info[f'agent{agent_id}_end_reason'] = 1  # crash
        success = False
        return done, success, info

    def _judge_overload(self, sim):
        flag_overload = False
        if sim.get_property_value(c.simulation_sim_time_sec) > 10:
            if (math.fabs(sim.get_property_value(c.accelerations_n_pilot_x_norm)) > self.acceleration_limit_x
                or math.fabs(sim.get_property_value(c.accelerations_n_pilot_y_norm)) > self.acceleration_limit_y
                or math.fabs(sim.get_property_value(c.accelerations_n_pilot_z_norm) + 1) > self.acceleration_limit_z
            ):
                flag_overload = True
        return flag_overload
