from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id=0, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = bool(env.sims[env.agent_names[agent_id]].get_property_value(c.detect_extreme_state))
        if done:
            print(f'INFO: [{env.agent_names[agent_id]}] is on an extreme state!')
            info[f'{env.agent_names[agent_id]}_end_reason'] = 1  # crash
        success = False
        return done, success, info
