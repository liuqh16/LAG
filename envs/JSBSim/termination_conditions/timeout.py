from .termination_condition_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super(Timeout, self).__init__(config)
        self.max_step = getattr(self.config, 'max_episode_steps', 500)

    def get_termination(self, task, env, robot_id=0):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success)
        """
        done = env.current_step >= self.max_step
        if done:
            print("INFO: Step limits!")
        success = False
        return done, success
