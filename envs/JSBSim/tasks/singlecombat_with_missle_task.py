import numpy as np
from collections import deque
from gym import spaces
from .singlecombat_task import SingleCombatTask
from ..reward_functions import AltitudeReward, MissileAttackReward, PostureReward, RelativeAltitudeReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..core.simulatior import MissileSimulator
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
        self.bloods, self.remaining_missiles, self.lock_duration = None, None, None

    def load_observation_space(self):
        self.observation_space = [spaces.Box(low=-10, high=10., shape=(18,)) for _ in range(self.num_agents)]

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = [spaces.MultiDiscrete([41, 41, 41, 30]) for _ in range(self.num_agents)]

    def normalize_observation(self, env, observations):
        return super().normalize_observation(env, observations)

    def normalize_action(self, env, actions):
        return super().normalize_action(env, actions)

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self.bloods = [100 for _ in range(self.num_aircrafts)]
        self.remaining_missiles = [env.aircraft_configs[uid].get("missile", 0) for uid in env.aircraft_configs.keys()]
        self.lock_duration = [deque(maxlen=int(1 / env.time_interval)) for _ in range(self.num_aircrafts)]
        return super().reset(env)

    def step(self, env, action):
        for agent_id in range(self.num_aircrafts):
            ego_idx, enm_idx = agent_id, (agent_id + 1) % self.num_aircrafts
            ego_uid, enm_uid = list(env.jsbsims.keys())[ego_idx], list(env.jsbsims.keys())[enm_idx]
            # [Rule-based missile launch]
            max_attack_angle = 22.5
            max_attack_distance = 6000
            target = env.jsbsims[enm_uid].get_position() - env.jsbsims[ego_uid].get_position()
            heading = env.jsbsims[ego_uid].get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) \
                / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[ego_idx].append(attack_angle < max_attack_angle)
            shoot_flag = np.sum(self.lock_duration[ego_idx]) >= self.lock_duration[ego_idx].maxlen \
                and distance <= max_attack_distance and self.remaining_missiles[ego_idx] > 0
            if shoot_flag:
                new_missile_uid = hex(int(env.jsbsims[ego_uid].uid, 16) + 1).lstrip("0x").upper()
                env.other_sims[new_missile_uid] = MissileSimulator.create(
                    parent=env.jsbsims[ego_uid],
                    target=env.jsbsims[enm_uid],
                    uid=new_missile_uid)
                self.remaining_missiles[ego_idx] -= 1
                print("Aircraft[{}] launched Missile[{}] --> target at Aircraft[{}]".format(
                    env.jsbsims[ego_uid].uid, new_missile_uid, env.jsbsims[enm_uid].uid
                ))
        active_sim_keys = list(env.other_sims.keys())
        for key in active_sim_keys:
            sim = env.other_sims[key]
            if isinstance(sim, MissileSimulator):
                if sim.is_success:
                    self.bloods[list(env.jsbsims.keys()).index(sim.target_aircraft.uid)] = 0
                elif sim.is_deleted:
                    env.other_sims.pop(sim.uid)

    def check_missile_warning(self, env, agent_id) -> MissileSimulator:
        ego_uid = list(env.jsbsims.keys())[agent_id]
        for sim in env.other_sims.values():
            if isinstance(sim, MissileSimulator):
                if sim.target_aircraft.uid == ego_uid:
                    return sim
        return None

    def get_termination(self, env, agent_id, info={}):
        done = False
        for condition in self.termination_conditions:
            d, _, info = condition.get_termination(self, env, agent_id, info)
            # force terminate if time is out
            if d and isinstance(condition, Timeout):
                return True, info
            # otherwise(ego crash) wait until there is no ego-missile alive
            elif done or d:
                for sim in env.other_sims.values():
                    if isinstance(sim, MissileSimulator) and sim.is_alive and \
                        sim._parent_uid == env.jsbsims[list(env.jsbsims.keys())[agent_id]].uid:
                        return False, info
            done = done or d
        return done, info
