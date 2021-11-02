from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class LowAltitude(BaseTerminationCondition):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)  # unit: m

    def get_termination(self, task, env, agent_id=0, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if altitude are too low.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = env.sims[agent_id].get_property_value(c.position_h_sl_m) <= self.altitude_limit
        if done:
            print(f'INFO: agent[{agent_id}] altitude is too low. Total Steps={env.current_step}')
            info[f'agent{agent_id}_end_reason'] = 1  # crash
        success = False
        return done, success, info
