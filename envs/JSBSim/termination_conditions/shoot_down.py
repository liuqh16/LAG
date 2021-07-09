from .termination_condition_base import BaseTerminationCondition


class ShootDown(BaseTerminationCondition):
    """
    ShootDown
    End up the simulation if the aircraft has been shot down.
    """

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 2500)  # unit: m

    def get_termination(self, task, env, agent_id=0, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft has been shot down.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = task.bloods[env.agent_names[agent_id]] <= 0
        if done:
            print(f'INFO: [{env.agent_names[agent_id]}] has been shot down!')
            info[f'{env.agent_names[agent_id]}_end_reason'] = 2  # shoot down
        success = False
        return done, success, info
