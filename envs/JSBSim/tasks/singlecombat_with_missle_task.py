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
        self.missile_lists = [Missile3D() for _ in range(self.num_agents)]

    def load_observation_space(self):
        self.observation_space = [spaces.Box(low=-10, high=10., shape=(18,)) for _ in range(self.num_agents)]
        # TODO: add extra obs_space for missile information here!

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]
        # TODO: if need to control missile launch mannully, add a new axes in self.action_space

    def normalize_observation(self, env, observations):
        return super().normalize_observation(env, observations)

    def normalize_action(self, env, actions):
        return super().normalize_action(env, actions)

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self.bloods = [100 for _ in range(self.num_agents)]
        self.missile_lists = [Missile3D() for _ in range(self.num_agents)]  # By default, both figher has 2 missiles.
        return super().reset(env)

    def step(self, env, action):
        for agent_id in range(self.num_agents):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_agents
            ego_feature = np.hstack([env.sims[ego_idx].get_position(), env.sims[ego_idx].get_velocity()])
            enm_feature = np.hstack([env.sims[enm_idx].get_position(), env.sims[enm_idx].get_velocity()])
            flag_hit = self.missile_lists[ego_idx].make_step(ego_feature, enm_feature, env.current_step)
            self.bloods[enm_idx] *= 1. - flag_hit

    def get_termination(self, env, agent_id, info={}):
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success or s
            # force terminate if time is out
            if d and isinstance(condition, Timeout):
                break
            # otherwise(ego crash) wait until there is no ego-missile alive
            elif done and self.missile_lists[agent_id].check_no_missile_alive():
                break
        return done, info
