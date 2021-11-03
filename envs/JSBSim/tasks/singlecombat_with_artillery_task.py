from .singlecombat_task import SingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, RelativeAltitudeReward, ArtilleryAttackReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout


class SingleCombatWithArtilleryTask(SingleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)

        self.reward_functions = [
            ArtilleryAttackReward(self.config),
            AltitudeReward(self.config),
            PostureReward(self.config),
            RelativeAltitudeReward(self.config),
        ]

        self.termination_conditions = [
            ShootDown(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

        self.bloods = [100 for _ in range(self.num_fighters)]

    def reset(self, env):
        """Task-specific reset, include reward function reset.

        Must call it after `env.get_observation()`
        """
        self.bloods = [100 for _ in range(self.num_fighters)]
        return super().reset(env)
