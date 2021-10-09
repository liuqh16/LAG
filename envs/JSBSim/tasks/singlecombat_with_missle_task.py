import numpy as np
from collections import OrderedDict
from gym import spaces
from .singlecombat_task import SingleCombatTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, MissileAttackReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..utils.missile_utils import Missile3D


class SingleCombatWithMissileTask(SingleCombatTask):
    def __init__(self, config: str):
        super().__init__(config)

        self.reward_functions = [
            MissileAttackReward(self.config),
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

        self.bloods = [100 for _ in range(self.num_agents)]
        self.num_missiles = getattr(self.config, 'num_missiles', 2)  # use this to set missile nums
        self.all_missiles = [Missile3D() for _ in range(self.num_fighters)]

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]
        # if need to control missile launch mannully, add a new axes in self.action_space

    def normalize_action(self, env, actions):
        """Convert discrete action index into continuous value.
        """
        def _normalize(action):
            action_norm = np.zeros(4)
            action_norm[0] = action[0] * 2. / (self.action_space[0].nvec[0] - 1.) - 1.
            action_norm[1] = action[1] * 2. / (self.action_space[0].nvec[1] - 1.) - 1.
            action_norm[2] = action[2] * 2. / (self.action_space[0].nvec[2] - 1.) - 1.
            action_norm[3] = action[3] * 0.5 / (self.action_space[0].nvec[3] - 1.) + 0.4
            return action_norm

        norm_act = np.zeros((self.num_fighters, 4))
        for agent_id in range(self.num_fighters):
            norm_act[agent_id] = _normalize(actions[agent_id])

        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.

        Must call it after `env.get_observation()`
        """
        self.bloods = [100 for _ in range(self.num_agents)]
        return super().reset(env)

    def step(self, env, action):
        # TODO: add missile step calculation here
        pass
