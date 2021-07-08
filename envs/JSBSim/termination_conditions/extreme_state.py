from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super(ExtremeState, self).__init__(config)

    def get_termination(self, task, env, agent_id=0):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success)
        """
        done = bool(env.sims[agent_id].get_property_value(c.detect_extreme_state))
        if done:
            print(f'INFO: [{task.agent_names[agent_id]}] is on an extreme state!')
        success = False
        return done, success
