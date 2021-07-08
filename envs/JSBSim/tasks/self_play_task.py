import math
import yaml
import os
import pdb
import numpy as np
from collections import OrderedDict
from gym import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..termination_conditions import ExtremeState, LowAltitude, Overload, ShootDown, Timeout
from ..utils.utils import lonlat2dis, get_AO_TA_R


act_space = spaces.Dict(
    OrderedDict({
        "aileron": spaces.Discrete(41),
        "elevator": spaces.Discrete(41),
        'rudder': spaces.Discrete(41),
        "throttle": spaces.Discrete(30),
    }))


obs_space = spaces.Dict(
    OrderedDict({
        'blue_fighter':
            spaces.Dict(
                OrderedDict({
                    'ego_info': spaces.Box(low=-10, high=10., shape=(22,)),
                })),
        'red_fighter':
            spaces.Dict(
                OrderedDict({
                    'ego_info': spaces.Box(low=-10, high=10., shape=(22, )),
                }))
    })
)


class SelfPlayTask(BaseTask):
    def __init__(self, config: str):
        super(SelfPlayTask, self).__init__(config)

        self.agent_names = list(self.config.init_config.keys())

        self.max_steps = self.config.max_steps
        self.reward_scale = self.config.reward_scale
        self.final_reward_scale = self.config.final_reward_scale
        self.danger_altitude = self.config.danger_altitude
        self.safe_altitude = self.config.safe_altitude
        self.target_dist = self.config.target_dist
        self.KL = self.config.KL
        self.Kv = self.config.Kv

        self.termination_conditions = [
            Timeout(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            ShootDown(self.config),
        ]

        self.bloods = {'blue_fighter': 100, 'red_fighter': 100}
        self.all_type_rewards = {'blue_fighter': None, 'red_fighter': None}
        self.pre_actions = None

    def init_variables(self):
        self.state_var = [
            c.position_long_gc_deg,                 # 1 / 10000   position_distance_from_start_lon_mt
            c.position_lat_geod_deg,                # 1 / 10000   position_distance_from_start_lat_mt
            c.position_h_sl_ft,                     # 0.304 / 5000
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
            c.velocities_v_north_fps,               # 0.304 / 340
            c.velocities_v_east_fps,                # 0.304 / 340
            c.velocities_v_down_fps,                # 0.304 / 340
            c.velocities_vc_fps,                    # 0.304 / 340
            c.accelerations_n_pilot_x_norm,         # 1 / 5
            c.accelerations_n_pilot_y_norm,         # 1 / 5
            c.accelerations_n_pilot_z_norm,         # 1 / 5
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,                 # [-1., 1.]    spaces.Discrete(11)
            c.fcs_elevator_cmd_norm,                # [-1., 1.]    spaces.Discrete(11)
            c.fcs_rudder_cmd_norm,                  # [-1., 1.]    spaces.Discrete(11)
            c.fcs_throttle_cmd_norm,                # [ 0., 1.]    spaces.Discrete(10)
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_ft,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad
        ]

    def init_conditions(self):
        self.init_position_conditions = {
            c.ic_long_gc_deg: 120.,     # 1.1  geodesic longitude [deg]
            c.ic_lat_geod_deg: 60.,     # 1.2  geodesic latitude  [deg]
        }
        self.init_condition = OrderedDict(
            [(agent, {
                c.ic_h_sl_ft: self.config.init_config[agent]['ic_h_sl_ft'],             # 1.1  altitude above mean sea level [ft]
                c.ic_terrain_elevation_ft: 0,                                           # +    default
                c.ic_long_gc_deg: self.config.init_config[agent]['ic_long_gc_deg'],     # 1.2  geodesic longitude [deg]
                c.ic_lat_geod_deg: self.config.init_config[agent]['ic_lat_geod_deg'],   # 1.3  geodesic latitude  [deg]
                c.ic_psi_true_deg: self.config.init_config[agent]['ic_psi_true_deg'],   # 5.   initial (true) heading [deg]   (0, 360)
                c.ic_u_fps: self.config.init_config[agent]['ic_u_fps'],                 # 2.1  body frame x-axis velocity [ft/s]  (-2200, 2200)
                c.ic_v_fps: 0,                                                          # 2.2  body frame y-axis velocity [ft/s]  (-2200, 2200)
                c.ic_w_fps: 0,                                                          # 2.3  body frame z-axis velocity [ft/s]  (-2200, 2200)
                c.ic_p_rad_sec: 0,                                                      # 3.1  roll rate  [rad/s]     (-2 * math.pi, 2 * math.pi)
                c.ic_q_rad_sec: 0,                                                      # 3.2  pitch rate [rad/s]     (-2 * math.pi, 2 * math.pi)
                c.ic_r_rad_sec: 0,                                                      # 3.3  yaw rate   [rad/s]     (-2 * math.pi, 2 * math.pi)
                c.ic_roc_fpm: 0,                                                        # 4.   initial rate of climb [ft/min]
                c.fcs_throttle_cmd_norm: 0.,                                            # 6.
            }) for agent in self.config.init_config.keys()]
        )
        self.aircraft_name = OrderedDict(
           [(agent, self.config.init_config[agent]['aircraft_name']) for agent in self.config.init_config.keys()]
        )

    def get_action_space(self):
        return act_space

    def get_observation_space(self):
        return obs_space

    def reset(self):
        self.bloods = {'blue_fighter': 100, 'red_fighter': 100}
        self.pre_actions = None

    def _obtain_fighter_observation_feature(self, sims):
        """
        blue_fighter: [pos_north_km, pos_east_km, pos_down_km, vel_north_mh, vel_east_mh, vel_down_mh]
        red_fighter: [pos_north_km, pos_east_km, pos_down_km, vel_north_mh, vel_east_mh, vel_down_mh]
        :param sims:
        :return:
        """
        init_b_longitude, init_b_latitude = self.init_position_conditions[c.ic_long_gc_deg], self.init_position_conditions[c.ic_lat_geod_deg]
        b_cur_longitude = sims[0].get_property_value(c.position_long_gc_deg)
        b_cur_latitude = sims[0].get_property_value(c.position_lat_geod_deg)
        b_cur_east, b_cur_north = lonlat2dis(b_cur_longitude, b_cur_latitude, init_b_longitude, init_b_latitude)
        b_cur_altitude = sims[0].get_property_value(c.position_h_sl_ft) * 0.304

        r_cur_longitude = sims[1].get_property_value(c.position_long_gc_deg)
        r_cur_latitude = sims[1].get_property_value(c.position_lat_geod_deg)
        r_cur_east, r_cur_north = lonlat2dis(r_cur_longitude, r_cur_latitude, init_b_longitude, init_b_latitude)
        r_cur_altitude = sims[1].get_property_value(c.position_h_sl_ft) * 0.304

        bv_north_mh = sims[0].get_property_value(c.velocities_v_north_fps) * 0.304 / 340
        bv_east_mh = sims[0].get_property_value(c.velocities_v_east_fps) * 0.304 / 340
        bv_down_mh = sims[0].get_property_value(c.velocities_v_down_fps) * 0.304 / 340
        rv_north_mh = sims[1].get_property_value(c.velocities_v_north_fps) * 0.304 / 340
        rv_east_mh = sims[1].get_property_value(c.velocities_v_east_fps) * 0.304 / 340
        rv_donw_mh = sims[1].get_property_value(c.velocities_v_down_fps) * 0.304 / 340
        blue_state = [b_cur_north / 1000, b_cur_east / 1000, b_cur_altitude / 1000, bv_north_mh, bv_east_mh, bv_down_mh]
        red_state = [r_cur_north / 1000, r_cur_east / 1000, r_cur_altitude / 1000, rv_north_mh, rv_east_mh, rv_donw_mh]
        return blue_state, red_state

    def get_reward(self, state, sims):
        # [north: km, east: km, down: km, v_n: mh, v_e: mh, v_d: mh]
        blue_state, red_state = self._obtain_fighter_observation_feature(sims)
        s_ori_b, s_ori_r = self._scoring_orientation_reward(blue_state, red_state)
        s_blood_b, s_blood_r = self._scoring_bloods_reward(blue_state, red_state)
        s_range = self._scoring_range_reward(blue_state, red_state)
        Pvh_b, Pvh_r = self._scoring_barrier_reward(blue_state, red_state, self.safe_altitude,
                                                    self.danger_altitude, Kv=self.Kv, KL=self.KL)
        self.all_type_rewards['blue_fighter'] = {'ori':s_ori_b, 'range': s_range, 'barrier': Pvh_b, 'blood':s_blood_b}
        self.all_type_rewards['red_fighter'] = {'ori':s_ori_r, 'range': s_range, 'barrier': Pvh_r, 'blood':s_blood_r}
        rewards = {'blue_reward': {'ori_range': s_ori_b * s_range, 'barrier': Pvh_b, 'blood': s_blood_b},
                   'red_reward': {'ori_range': s_ori_r * s_range, 'barrier': Pvh_r, 'blood': s_blood_r}}
        return rewards

    def reward_for_smooth_action(self, action_dicts=None):
        if self.pre_actions is None:
            self.pre_actions = {"red_fighter": np.array([20., 18.6, 20., 0.]),
                                'blue_fighter': np.array([20., 18.6, 20., 0.])}
            return
        cur_blue_action, pre_blue_action = action_dicts['blue_fighter'], self.pre_actions['blue_fighter']
        cur_red_action, pre_red_action = action_dicts['red_fighter'], self.pre_actions['red_fighter']
        delta_blue_action = np.abs(cur_blue_action - pre_blue_action)
        delta_red_action = np.abs(cur_red_action - pre_red_action)
        delta_blue_action = np.mean(delta_blue_action * (delta_blue_action > 10)) * 0.001
        delta_red_action = np.mean(delta_red_action * (delta_red_action > 10)) * 0.001
        return {'blue_reward': {'smooth_act': -delta_blue_action}, 'red_reward': {'smooth_act': -delta_red_action}}

    def _update_bloods(self, BA, AA, distance, fighter_type):
        delta_blood = 0
        if np.abs(np.rad2deg(AA)) < 60 and np.abs(np.rad2deg(BA)) < 30 and distance <= 3:
            delta_blood = -10 * 0
        self.bloods[fighter_type] += delta_blood
        return delta_blood

    def _scoring_bloods_reward(self, blue_state, red_state):
        # [north: km, east: km, down: km, v_n: mh, v_e: mh, v_d: mh]
        BA_b, AA_b, distance = get_AO_TA_R(blue_state, red_state)
        BA_r, AA_r, distance = get_AO_TA_R(red_state, blue_state)
        delta_b_blood = self._update_bloods(BA_b, AA_b, distance, 'blue_fighter')
        delta_r_blood = self._update_bloods(BA_r, AA_r, distance, 'red_fighter')
        scoring_b_blood = -delta_r_blood / 1. * 0.
        scoring_r_blood = -delta_b_blood / 1. * 0.
        return scoring_b_blood, scoring_r_blood

    def _scoring_orientation_reward(self, blue_state, red_state):
        # [north: km, east: km, down: km, v_n: mh, v_e: mh, v_d: mh]
        # orientaion reward --> [0, 1]
        BA_b, AA_b, distance = get_AO_TA_R(blue_state, red_state)
        BA_r, AA_r, distance = get_AO_TA_R(red_state, blue_state)

        score_enm_b_ori = min((np.arctanh(1. - max(2 * AA_b / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        # score_ego_b_ori = (1. - np.tanh(9 * (BA_b - np.pi / 9))) / 3. + 1 / 3.
        score_ego_b_ori = 1 / (50 * BA_b / np.pi + 2) + 1 / 2
        scoring_b_ori = score_enm_b_ori + score_ego_b_ori

        score_enm_r_ori = min((np.arctanh(1. - max(2 * AA_r / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        # score_ego_r_ori = (1. - np.tanh(9 * (BA_r - np.pi / 9))) / 3. + 1 / 3.
        score_ego_r_ori = 1 / (50 * BA_r / np.pi + 2) + 1 / 2
        scoring_r_ori = score_enm_r_ori + score_ego_r_ori
        return scoring_b_ori, scoring_r_ori

    def _scoring_orientation_reward_basic(self, blue_state, red_state):
        # [north: km, east: km, down: km, v_n: mh, v_e: mh, v_d: mh]
        # orientaion reward --> [0, 1]
        BA_b, AA_b, distance = get_AO_TA_R(blue_state, red_state)
        BA_r, AA_r, distance = get_AO_TA_R(red_state, blue_state)

        score_enm_b_ori = min((np.arctanh(1. - max(2 * AA_b / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        score_ego_b_ori = (1. - np.tanh(9 * (BA_b - np.pi / 9))) / 3. + 1 / 3.
        scoring_b_ori = score_enm_b_ori + score_ego_b_ori

        score_enm_r_ori = min((np.arctanh(1. - max(2 * AA_r / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        score_ego_r_ori = (1. - np.tanh(9 * (BA_r - np.pi / 9))) / 3. + 1 / 3.
        scoring_r_ori = score_enm_r_ori + score_ego_r_ori
        return scoring_b_ori, scoring_r_ori

    def _scoring_orientation_reward_old(self, blue_state, red_state):
        # [north: km, east: km, down: km, v_n: mh, v_e: mh, v_d: mh]
        # orientaion reward --> [0, 1]
        BA_b, AA_b, distance = get_AO_TA_R(blue_state, red_state)
        BA_r, AA_r, distance = get_AO_TA_R(red_state, blue_state)

        score_enm_b_ori = (np.arctanh(1. - max(2 * AA_b / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        score_ego_b_ori = (1. - np.tanh(2 * (BA_b - np.pi / 2))) / 2.
        scoring_b_ori = score_enm_b_ori * score_ego_b_ori

        score_enm_r_ori = (np.arctanh(1. - max(2 * AA_r / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        score_ego_r_ori = (1. - np.tanh(2 * (BA_r - np.pi / 2))) / 2.
        scoring_r_ori = score_enm_r_ori * score_ego_r_ori
        return scoring_b_ori, scoring_r_ori

    def _scoring_range_reward(self, blue_state, red_state):
        bx, by, bz = blue_state[:3]
        rx, ry, rz = red_state[:3]
        dist = np.linalg.norm([bx - rx, by - ry, bz - rz])
        scoring_range = np.clip(1.2 * np.min([np.exp(-(dist - self.target_dist) * 0.21), 1]) / (1. + np.exp(-(dist - self.target_dist + 1)*0.8)), 0.3, 1)
        # scoring_range = np.exp(-(dist - env_config.target_dist) ** 2 * 0.004) / (1. + np.exp(-(dist - env_config.target_dist + 2) * 2))
        return scoring_range

    def _scoring_barrier_reward(self, blue_state, red_state, safe_altitude, danger_altitude, Kv, KL):
        # [north: km, east: km, down: km, v_n: mh, v_e: mh, v_d: mh]
        bx, by, bz, bvx, bvy, bvz = blue_state
        rx, ry, rz, rvx, rvy, rvz = red_state
        # 1) Punishment of velocity when lower than safe altitude   (range: [-1, 0.001])
        b_Pv, r_Pv = 0., 0.
        if bz <= safe_altitude:
            b_Pv = - np.clip(bvz / Kv * (safe_altitude - bz) / safe_altitude, -0.001, 1)
        if rz <= safe_altitude:
            r_Pv = - np.clip(rvz / Kv * (safe_altitude - rz) / safe_altitude, -0.001, 1)
        # 2) Punishment of altitude when lower than danger altitude (range: [-1, 0])
        b_PH, r_PH = 0., 0.
        if bz <= danger_altitude:
            # when the altitude is too low, we must ignore the orientation's advantage and merely focus on altitude.
            b_PH = np.clip(bz / danger_altitude, 0., 1.) - 1. - 1.
        if rz <= danger_altitude:
            # when the altitude is too low, we must ignore the orientation's advantage and merely focus on altitude.
            r_PH = np.clip(rz / danger_altitude, 0., 1.) - 1. - 1.
        # 3) Punishment of relative altitude when larger than 1000  (range: [-1, 0])
        b_PLr, r_PLb = 0., 0.
        if bz - rz < -KL:
            b_PLr = bz - rz + KL
        elif rz - bz < -KL:
            r_PLb = rz - bz + KL

        if bz - rz > KL:
            b_PLr = KL - (bz - rz)
        elif rz - bz > KL:
            r_PLb = KL - (rz - bz)
        return b_Pv + b_PH + b_PLr, r_Pv + r_PH + r_PLb
