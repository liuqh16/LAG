from .termination_condition_base import BaseTerminationCondition


class ShootDown(BaseTerminationCondition):
    """
    ShootDown
    End up the simulation if the aircraft has been shot down.
    """

    def __init__(self, config):
        super(ShootDown, self).__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)  # unit: m

    def get_termination(self, task, env, agent_id=0):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft has been shot down.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success)
        """
        done = task.bloods[task.agent_names[agent_id]] <= 0
        if done:
            print(f'INFO: [{task.agent_names[agent_id]}] has been shot down!')
        success = False
        return done, success
