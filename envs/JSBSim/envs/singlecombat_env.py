import numpy as np
from collections import OrderedDict
from .env_base import BaseEnv
from ..core.catalog import Catalog
from ..core.simulation import Simulation
from ..tasks.singlecombat_task import SingleCombatTask
from ..tasks.singlecombat_with_missle_task import SingleCombatWithMissileTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config: str):
        super().__init__(config)

        self.aircraft_names = [self.config.init_config[idx]['aircraft_name'] for idx in range(self.num_agents)]

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'singlecombat_with_missile':
            self.task = SingleCombatWithMissileTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")
        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

    def reset(self):
        self.current_step = 0
        self.close()
        self.reset_conditions()
        # recreate simulation
        self.sims = [Simulation(
            aircraft_name=self.aircraft_names[idx],
            init_conditions=self.init_conditions[idx],
            origin_point=(self.init_longitude, self.init_latitude),
            jsbsim_freq=self.jsbsim_freq,
            agent_interaction_steps=self.agent_interaction_steps) for idx in range(self.num_agents)]
        next_observation = self.get_observation()
        self.task.reset(self)
        return next_observation

    def reset_conditions(self):
        # TODO: randomization
        # Origin point of Combat Field [geodesic longitude&latitude (deg)]
        self.init_longitude, self.init_latitude = 120.0, 60.0
        # Initial setting of each agent
        self.init_conditions = [{
            Catalog.ic_h_sl_ft: self.config.init_config[idx]['ic_h_sl_ft'],             # 1.1  altitude above mean sea level [ft]
            Catalog.ic_terrain_elevation_ft: 0,                                         # +    default
            Catalog.ic_long_gc_deg: self.config.init_config[idx]['ic_long_gc_deg'],     # 1.2  geodesic longitude [deg]
            Catalog.ic_lat_geod_deg: self.config.init_config[idx]['ic_lat_geod_deg'],   # 1.3  geodesic latitude  [deg]
            Catalog.ic_psi_true_deg: self.config.init_config[idx]['ic_psi_true_deg'],   # 5.   initial (true) heading [deg]   (0, 360)
            Catalog.ic_u_fps: self.config.init_config[idx]['ic_u_fps'],                 # 2.1  body frame x-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_v_fps: 0,                                                        # 2.2  body frame y-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_w_fps: 0,                                                        # 2.3  body frame z-axis velocity [ft/s]  (-2200, 2200)
            Catalog.ic_p_rad_sec: 0,                                                    # 3.1  roll rate  [rad/s]     (-2 * math.pi, 2 * math.pi)
            Catalog.ic_q_rad_sec: 0,                                                    # 3.2  pitch rate [rad/s]     (-2 * math.pi, 2 * math.pi)
            Catalog.ic_r_rad_sec: 0,                                                    # 3.3  yaw rate   [rad/s]     (-2 * math.pi, 2 * math.pi)
            Catalog.ic_roc_fpm: 0,                                                      # 4.   initial rate of climb [ft/min]
            Catalog.fcs_throttle_cmd_norm: 0.,                                          # 6.
        } for idx in range(self.num_agents)]

    def step(self, actions: list):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and 
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.array): the agents' action, with same length as action space.

        Returns:
            (tuple):
                state: agent's observation of the current environment
                reward_visualize: amount of reward_visualize returned after previous action
                done: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {}

        actions = self.task.normalize_action(self, actions)
        next_observation = self.make_step(actions)

        rewards = np.zeros(self.num_agents)
        for agent_id in range(self.num_agents):
            rewards[agent_id], info = self.task.get_reward(self, agent_id, info)

        done = False
        for agent_id in range(self.num_agents):
            agent_done, info = self.task.get_termination(self, agent_id, info)
            done = agent_done or done
        dones = done * np.ones(self.num_agents)

        return next_observation, rewards, dones, info

    def get_observation(self):
        """
        get state observation from sim.

        Returns:
            (OrderedDict): the same format as self.observation_space
        """
        next_observation = []
        for agent_id in range(self.num_agents):
            next_observation.append(self.sims[agent_id].get_property_values(self.task.state_var))
        next_observation = self.task.normalize_observation(self, next_observation)
        return next_observation

    def close(self):
        """Cleans up this environment's objects.

        Environments automatically close() when garbage collected or when the program exits.
        """
        for agent_id in range(self.num_agents):
            if self.sims[agent_id]:
                self.sims[agent_id].close()

    def render(self):
        # TODO: real time rendering
        render_list = []
        for agent_id in range(self.num_agents):
            render_list.append(np.array(self.sims[agent_id].get_property_values(self.task.render_var)))
        return np.hstack(render_list)

    def seed(self, seed):
        # TODO: random seed
        return super().seed(seed=seed)
