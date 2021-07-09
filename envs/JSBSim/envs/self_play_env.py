import numpy as np
from collections import OrderedDict
from .env_base import BaseEnv
from ..core.catalog import Catalog as c
from ..core.simulation import Simulation
from ..tasks.self_play_task import act_space, SelfPlayTask
from ..utils.utils import lonlat2dis


class JSBSimSelfPlayEnv(BaseEnv):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "csv"]}

    def __init__(self, config='r_hyperparams.yaml'):
        self.sims = None
        self.task = SelfPlayTask(config)
        self.config = self.task.config
        self.observation_space = self.task.get_observation_space()
        self.action_space = self.task.get_action_space()

        # nickname of each agent
        self.agent_names = list(self.config.init_config.keys())
        # aircraft model of each agent
        self.aircraft_name = OrderedDict(
           [(agent, self.config.init_config[agent]['aircraft_name']) for agent in self.config.init_config.keys()]
        )
        self.num_agents = len(self.agent_names)
        self.max_steps = self.config.max_steps
        # set controlling frequency
        self.jsbsim_freq = self.config.jsbsim_freq
        self.agent_interaction_steps = self.config.agent_interaction_steps

        self.current_step = 0
        self.actions = OrderedDict([(agent, np.zeros(len(self.task.action_var))) for agent in self.agent_names])
        self.features = OrderedDict([(agent, np.zeros(len(self.task.feature_var))) for agent in self.agent_names])

    def reset(self):
        self.current_step = 0
        # Default action is straight forward
        self.actions = OrderedDict([(agent, np.array([20., 18.6, 20., 0.])) for agent in self.agent_names])
        self.close()
        self.reset_conditions()
        # recreate simulation
        self.sims = OrderedDict([(agent, Simulation(
            aircraft_name=self.aircraft_name[agent],
            init_conditions=self.init_condition[agent],
            jsbsim_freq=self.jsbsim_freq,
            agent_interaction_steps=self.agent_interaction_steps)) for agent in self.agent_names])
        next_observation = self.get_observation()
        self.task.reset(self)
        return next_observation

    def reset_conditions(self):
        # TODO: randomization
        # Origin point of Combat Field [geodesic longitude&latitude (deg)]
        self.init_longitude, self.init_latitude = 120.0, 60.0
        # Initial setting of each agent
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

    def step(self, action: dict):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and 
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (dict{str: np.array}): the agents' action, with same length as action variables.

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward_visualize: amount of reward_visualize returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {}
        self.actions = self.process_actions(action)
        self.make_step(self.actions)
        next_observation = self.get_observation()
        reward = OrderedDict()
        for agent_id, agent_name in enumerate(self.agent_names):
            reward[agent_name], info = self.task.get_reward(self, agent_id, info)
        done = False
        for agent_id in range(self.num_agents):
            agent_done, info = self.task.get_termination(self, agent_id, info)
            info[f'{self.agent_names[agent_id]}_done'] = agent_done
            done = agent_done or done

        return next_observation, reward, done, info

    def make_step(self, action: dict):
        """
        Calculates new state.

        Args:
            action (dict{str: np.array}): the agents' action, with same length as action variables.
        """
        for agent_name in self.agent_names:
            # take actions
            self.sims[agent_name].set_property_values(self.task.action_var, action[agent_name])
            # run simulation
            self.sims[agent_name].run()

    def get_observation(self):
        """
        get state observation from sim.

        Returns:
            (OrderedDict): the same format as self.observation_space
        """
        # generate observation (gym.Env output)
        all_obs_list = []
        for agent_name in self.agent_names:
            all_obs_list.append(self.sims[agent_name].get_property_values(self.task.state_var))
        next_observation = OrderedDict()
        for (agent_id, agent_name) in enumerate(self.agent_names):
            obs_norm = self.normalize_observation(all_obs_list[agent_id:] + all_obs_list[:agent_id])
            next_observation[agent_name] = OrderedDict({'ego_info': obs_norm})
        # generate feature (use in reward calculation)
        for agent_name in self.agent_names:
            # unit: (degree, degree, ft, fps, fps, fps)
            lat, lon, alt, vn, ve, vd = self.sims[agent_name].get_property_values(self.task.feature_var)
            # unit: degree -> m
            east, north = lonlat2dis(lon, lat, self.init_longitude, self.init_latitude)
            # unit: (km, km, km, mh, mh, mh)
            self.features[agent_name][0:3] = np.array([east, north, alt * 0.304]) / 1000
            self.features[agent_name][3:6] = np.array([vn, ve, vd]) * 0.304 / 340
        return next_observation

    def normalize_observation(self, sorted_obs_list):
        ego_obs_list, enm_obs_list = sorted_obs_list[0], sorted_obs_list[1]
        observation = np.zeros(22)
        ego_cur_east, ego_cur_north = lonlat2dis(ego_obs_list[0], ego_obs_list[1], self.init_longitude, self.init_latitude)
        enm_cur_east, enm_cur_north = lonlat2dis(enm_obs_list[0], enm_obs_list[1], self.init_longitude, self.init_latitude)
        observation[0] = ego_cur_north / 10000.
        observation[1] = ego_cur_east / 10000.
        observation[2] = ego_obs_list[2] * 0.304 / 5000
        observation[3] = np.cos(ego_obs_list[3])
        observation[4] = np.sin(ego_obs_list[3])
        observation[5] = np.cos(ego_obs_list[4])
        observation[6] = np.sin(ego_obs_list[4])
        observation[7] = np.cos(ego_obs_list[5])
        observation[8] = np.sin(ego_obs_list[5])
        observation[9] = ego_obs_list[6] * 0.304 / 340
        observation[10] = ego_obs_list[7] * 0.304 / 340
        observation[11] = ego_obs_list[8] * 0.304 / 340
        observation[12] = ego_obs_list[9] * 0.304 / 340
        observation[13] = ego_obs_list[10] / 5
        observation[14] = ego_obs_list[11] / 5
        observation[15] = ego_obs_list[12] / 5
        observation[16] = enm_cur_north / 10000.
        observation[17] = enm_cur_east / 10000.
        observation[18] = enm_obs_list[2] * 0.304 / 5000
        observation[19] = enm_obs_list[6] * 0.304 / 340
        observation[20] = enm_obs_list[7] * 0.304 / 340
        observation[21] = enm_obs_list[8] * 0.304 / 340
        return observation

    def process_actions(self, action: dict):
        for agent_name in self.agent_names:
            action[agent_name] = np.array(action[agent_name], dtype=np.float32)
            action[agent_name][0] = action[agent_name][0] * 2. / (act_space['aileron'].n - 1.) - 1.
            action[agent_name][1] = action[agent_name][1] * 2. / (act_space['elevator'].n - 1.) - 1.
            action[agent_name][2] = action[agent_name][2] * 2. / (act_space['rudder'].n - 1.) - 1.
            action[agent_name][3] = action[agent_name][3] * 0.5 / (act_space['throttle'].n - 1.) + 0.4
        return action

    def close(self):
        """Cleans up this environment's objects.

        Environments automatically close() when garbage collected or when the program exits.
        """
        if self.sims:
            for agent_name in self.agent_names:
                self.sims[agent_name].close()

    def render(self, mode="human", **kwargs):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - csv: output to cvs files

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        obs_list = []
        for agent_name in self.agent_names:
            obs_list.append(np.array(self.sims[agent_name].get_property_values(self.task.render_var)))
        return np.hstack(obs_list)
