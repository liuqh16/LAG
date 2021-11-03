import numpy as np
from collections import OrderedDict
from gym import spaces
from .singlecombat_task import SingleCombatTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, MissileAttackReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..utils.missile_utils import Missile3D
from ..utils.utils import NEU2LLA


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
        self.init_missile()

    def init_missile(self):
        self.bloods = [100 for _ in range(self.num_aircrafts)]
        self.missile_lists = [Missile3D() for _ in range(self.num_aircrafts)] # By default, both figher has 1 missile.

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
        self.init_missile()
        return super().reset(env)

    def step(self, env, action):
        for agent_id in range(self.num_aircrafts):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_aircrafts
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

    def get_missile_trajectory(self, env, agent_id):
        """
        render missile trajectory, if no missile, use default value instead
        """
        missile_render = []
        for i in range(self.missile_lists[agent_id].num_missile):
            missile_state = np.zeros((6,))
            if self.missile_lists[agent_id].missile_info[i]['launched']:
                missile_state[:3] = np.array(self.missile_lists[agent_id].missile_info[i]['current_state'][:3])
                missile_state[4:6] = np.array(self.missile_lists[agent_id].missile_info[i]['current_state'][4:6])[::-1]
            else:
                missile_state[:3] = env.sims[agent_id].get_position()
                missile_state[4:6] = env.sims[agent_id].get_rpy()[1:]
            # unit conversion (m, m, m) => (degree, degree, ft)
            missile_state[:3] = NEU2LLA(*missile_state[:3], env.center_lon, env.center_lat, env.center_alt)
            missile_render.append(missile_state)
        return missile_render


class SingleCombatWithAvoidMissileTask(SingleCombatWithMissileTask):
    def init_missile(self):
        self.bloods = [100 for _ in range(self.num_aircrafts)]
        self.missile_lists = [Missile3D(allow_shoot=False), Missile3D(allow_shoot=True)] # By default, both figher has 1 missile.