from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -100
    - Crash accidentally: -200
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        from ..envs.singlecombat_env import SingleCombatEnv
        assert isinstance(env, SingleCombatEnv)
        reward = 0
        if env.agents[agent_id].is_shotdown:
            reward -= 100
        elif env.agents[agent_id].is_crash:
            reward -= 200
        return self._process(reward, agent_id)
