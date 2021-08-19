import numpy as np
from collections import OrderedDict
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
from ..utils.utils import lonlat2dis

# TODO: define observation
class HeadingTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
        ]

        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

        self.init_longitude, self.init_latitude = 0, 0

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: feet)
            c.delta_heading,                    #  1.           (unit: degree)
            c.attitude_roll_rad,                #  2. roll      (unit: rad)
            c.attitude_pitch_rad,               #  3. pitch     (unit: rad)
            c.velocities_v_north_fps,           #  4. v_north   (unit: fps)
            c.velocities_v_east_fps,            #  5. v_east    (unit: fps)
            c.velocities_v_down_fps,            #  6. v_down    (unit: fps)
            c.velocities_vc_fps,                #  7. vc        (unit: fps)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_ft,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        space_dict = OrderedDict()
        for fighter_name in list(self.config.init_config.keys()):
            space_dict[fighter_name] = spaces.Dict(
                        OrderedDict({
                            'ego_info': spaces.Box(low=-10, high=10., shape=(8,)), #  NOTE: shape 
                        }))
        self.observation_space = spaces.Dict(space_dict)

    def load_action_space(self):
        space_dict = OrderedDict({
                "aileron": spaces.Discrete(41),
                "elevator": spaces.Discrete(41),
                'rudder': spaces.Discrete(41),
                "throttle": spaces.Discrete(30),
            })
        self.action_space = spaces.Dict(space_dict)

    def normalize_observation(self, env, sorted_all_obs_list: list):
        ego_obs_list = sorted_all_obs_list[0]
        observation = np.zeros(8)
        observation[0] = ego_obs_list[0] * 0.304 / 1000     #  0. ego delta altitude  (unit: 1km)
        observation[1] = ego_obs_list[1] / 180 * np.pi      #  1. ego delta heading   (unit rad)
        observation[2] = ego_obs_list[2]                    #  2. ego_roll    (unit: rad)
        observation[3] = ego_obs_list[3]                    #  3. ego_pitch   (unit: rad)
        observation[4] = ego_obs_list[4] * 0.304 / 340      #  4. ego_v_north        (unit: mh)
        observation[5] = ego_obs_list[5] * 0.304 / 340      #  5. ego_v_east        (unit: mh)
        observation[6] = ego_obs_list[6] * 0.304 / 340      #  6. ego_v_down        (unit: mh)
        observation[7] = ego_obs_list[7] * 0.304 / 340      #  7. ego_vc        (unit: mh)

        return observation

    def process_actions(self, env, action: dict):
        """Convert discrete action index into continuous value.
        """
        for agent_name in env.agent_names:
            action[agent_name] = np.array(action[agent_name], dtype=np.float32)
            action[agent_name][0] = action[agent_name][0] * 2. / (self.action_space['aileron'].n - 1.) - 1.
            action[agent_name][1] = action[agent_name][1] * 2. / (self.action_space['elevator'].n - 1.) - 1.
            action[agent_name][2] = action[agent_name][2] * 2. / (self.action_space['rudder'].n - 1.) - 1.
            action[agent_name][3] = action[agent_name][3] * 0.5 / (self.action_space['throttle'].n - 1.) + 0.4
        return action

    def reset(self, env):
        """Task-specific reset, include reward function reset.

        Must call it after `env.get_observation()`
        """
        return super().reset(env)

    def get_reward(self, env, agent_id, info={}):
        """
        Must call it after `env.get_observation()`
        """
        return super().get_reward(env, agent_id, info)

    def get_termination(self, env, agent_id, info={}):
        return super().get_termination(env, agent_id, info)
