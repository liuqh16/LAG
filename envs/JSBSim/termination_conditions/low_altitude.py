from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class LowAltitude(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """

    def __init__(self, config):
        super(LowAltitude, self).__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)  # unit: m

    def get_termination(self, task, env, agent_id=0):
        """
        Return whether the episode should terminate.
        End up the simulation if altitude are too low.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success)
        """
        done = env.sims[agent_id].get_property_value(c.position_h_sl_ft) <= self.altitude_limit * (1 / 0.3048)
        if done:
            print(f'INFO: the {task.agent_names[agent_id]} altitude are too low')
        success = False
        return done, success
