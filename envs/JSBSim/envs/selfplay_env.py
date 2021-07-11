import numpy as np
from collections import OrderedDict
from .env_base import BaseEnv
from ..core.catalog import Catalog
from ..core.simulation import Simulation
from ..tasks.selfplay_task import SelfPlayTask
from ..tasks.selfplay_with_missle_task import SelfPlayWithMissileTask
from ..utils.utils import lonlat2dis


class SelfPlayEnv(BaseEnv):
    """
    SelfPlayEnv is an one-to-one competitive environment.
    """
    metadata = {"render.modes": ["human", "csv"]}

    def __init__(self, config='selfplay_task.yaml'):
        if 'missile' not in config:
            self.task = SelfPlayTask(config)
        else:
            self.task = SelfPlayWithMissileTask(config)
        super().__init__(self.task, config)

    def init_variables(self):
        self.current_step = 0
        self.sims = OrderedDict([(agent, None) for agent in self.agent_names])
        self.init_longitude, self.init_latitude = 0.0, 0.0
        self.init_conditions = OrderedDict([(agent, None) for agent in self.agent_names])
        self.actions = OrderedDict([(agent, np.zeros(len(self.task.action_var))) for agent in self.agent_names])

    def reset(self):
        self.current_step = 0
        # Default action is straight forward
        self.actions = OrderedDict([(agent, np.array([20., 18.6, 20., 0.])) for agent in self.agent_names])
        self.close()
        self.reset_conditions()
        # recreate simulation
        self.sims = OrderedDict([(agent, Simulation(
            aircraft_name=self.aircraft_names[agent],
            init_conditions=self.init_conditions[agent],
            origin_point=(self.init_longitude, self.init_latitude),
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
        self.init_conditions = OrderedDict(
            [(agent, {
                Catalog.ic_h_sl_ft: self.config.init_config[agent]['ic_h_sl_ft'],             # 1.1  altitude above mean sea level [ft]
                Catalog.ic_terrain_elevation_ft: 0,                                           # +    default
                Catalog.ic_long_gc_deg: self.config.init_config[agent]['ic_long_gc_deg'],     # 1.2  geodesic longitude [deg]
                Catalog.ic_lat_geod_deg: self.config.init_config[agent]['ic_lat_geod_deg'],   # 1.3  geodesic latitude  [deg]
                Catalog.ic_psi_true_deg: self.config.init_config[agent]['ic_psi_true_deg'],   # 5.   initial (true) heading [deg]   (0, 360)
                Catalog.ic_u_fps: self.config.init_config[agent]['ic_u_fps'],                 # 2.1  body frame x-axis velocity [ft/s]  (-2200, 2200)
                Catalog.ic_v_fps: 0,                                                          # 2.2  body frame y-axis velocity [ft/s]  (-2200, 2200)
                Catalog.ic_w_fps: 0,                                                          # 2.3  body frame z-axis velocity [ft/s]  (-2200, 2200)
                Catalog.ic_p_rad_sec: 0,                                                      # 3.1  roll rate  [rad/s]     (-2 * math.pi, 2 * math.pi)
                Catalog.ic_q_rad_sec: 0,                                                      # 3.2  pitch rate [rad/s]     (-2 * math.pi, 2 * math.pi)
                Catalog.ic_r_rad_sec: 0,                                                      # 3.3  yaw rate   [rad/s]     (-2 * math.pi, 2 * math.pi)
                Catalog.ic_roc_fpm: 0,                                                        # 4.   initial rate of climb [ft/min]
                Catalog.fcs_throttle_cmd_norm: 0.,                                            # 6.
            }) for agent in self.agent_names]
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
            action[agent_name][0] = action[agent_name][0] * 2. / (self.action_space['aileron'].n - 1.) - 1.
            action[agent_name][1] = action[agent_name][1] * 2. / (self.action_space['elevator'].n - 1.) - 1.
            action[agent_name][2] = action[agent_name][2] * 2. / (self.action_space['rudder'].n - 1.) - 1.
            action[agent_name][3] = action[agent_name][3] * 0.5 / (self.action_space['throttle'].n - 1.) + 0.4
        return action

    def close(self):
        """Cleans up this environment's objects.

        Environments automatically close() when garbage collected or when the program exits.
        """
        for agent_name in self.agent_names:
            if self.sims[agent_name]:
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
