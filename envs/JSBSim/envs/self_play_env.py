import numpy as np
from ..core.catalog import Catalog as c
from collections import OrderedDict
from .env_base import BaseEnv
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
        self.max_episode_steps = self.task.max_episode_steps
        self.observation_space = self.task.get_observation_space()  # None
        self.action_space = self.task.get_action_space()  # None

        self.state = None
        self.pre_reward_obs = None
        self.cur_step = 0
        self.end_reason = {'blue_crash': False, 'red_crash': False}
        self.trajectory = []

    def reset(self):
        self.cur_step = 0
        self.trajectory = []
        self.end_reason = {'blue_crash': False, 'red_crash': False}
        self.task.pre_actions = None
        self.task.bloods = {'blue_fighter': 100, 'red_fighter': 100}
        self.close()
        self.sims = [Simulation(
            aircraft_name=self.task.aircraft_names[agent],
            init_conditions=self.task.init_condition[agent],
            jsbsim_freq=self.task.jsbsim_freq,
            agent_interaction_steps=self.task.agent_interaction_steps) for agent in self.task.agent_names]
        obs_b, obs_r = self.get_observation()
        blue_obs = OrderedDict({'ego_info': obs_b})
        red_obs = OrderedDict({'ego_info': obs_r})
        self.pre_reward_obs = self.task.get_reward(self.state, self.sims)
        self.task.reward_for_smooth_action()
        return OrderedDict({'blue_fighter': blue_obs, 'red_fighter': red_obs})

    def step(self, action_dicts):
        """

        Run one timestep of the environment's dynamics. When end of

        episode is reached, you are responsible for calling `reset()`

        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward_visualize, done, info).



        :param action_dicts: np.array, the agent's action, with same length as action variables.

        :return:

            state: agent's observation of the current environment

            reward_visualize: amount of reward_visualize returned after previous action

            done: whether the episode has ended, in which case further step() calls are undefined

            info: auxiliary information

        """
        self.cur_step += 1
        blue_action = self.process_actions(action_dicts, 'blue_fighter')
        red_action = self.process_actions(action_dicts, 'red_fighter')
        if blue_action is not None:
            if not len(blue_action) == len(self.action_space.spaces):
                raise ValueError("mismatch between action and action space size")
        self.make_step([blue_action, red_action])
        obs_b, obs_r = self.get_observation()
        blue_obs = OrderedDict({'ego_info': obs_b})
        red_obs = OrderedDict({'ego_info': obs_r})
        next_observation = OrderedDict({'blue_fighter': blue_obs, 'red_fighter': red_obs})
        reward = dict()
        cur_reward_obs_dicts = self.task.get_reward(self.state, self.sims)
        cur_reward_act_dicts = self.task.reward_for_smooth_action(action_dicts)
        cur_reward_b, cur_reward_r = cur_reward_obs_dicts['blue_reward'], cur_reward_obs_dicts['red_reward']
        reward['blue_reward'] = (cur_reward_b['ori_range'] - self.pre_reward_obs['blue_reward']['ori_range']) * self.task.reward_scale
        reward['blue_reward'] += cur_reward_b['barrier'] + cur_reward_b['blood'] + cur_reward_act_dicts['blue_reward']['smooth_act']
        reward['red_reward'] = (cur_reward_r['ori_range'] - self.pre_reward_obs['red_reward']['ori_range']) * self.task.reward_scale
        reward['red_reward'] += cur_reward_r['barrier'] + cur_reward_r['blood'] + cur_reward_act_dicts['red_reward']['smooth_act']
        self.pre_reward_obs = cur_reward_obs_dicts

        done = self.is_terminal()
        sign = self._judge_terminal_condition(crash_done=done)
        if self.cur_step >= self.max_episode_steps:
            print('steps limits')
            done = True
        reward['blue_final_reward'] = sign * self.task.final_reward_scale
        reward['red_final_reward'] = -sign * self.task.final_reward_scale

        return next_observation, reward, done, self.end_reason

    def _judge_terminal_condition(self, crash_done):
        if crash_done is False:
            return 0.
        sign = 0.
        if self.end_reason['blue_fighter_crash'] and self.end_reason['red_fighter_crash']:
            sign = 0.
        elif self.end_reason['blue_fighter_crash'] and self.task.bloods['blue_fighter'] <= 0:
            sign = -1
        elif self.end_reason['red_fighter_crash'] and self.task.bloods['red_fighter'] <= 0:
            sign = 1
        self.end_reason['blue_win'] = (1. + sign) / 2.
        self.end_reason['red_win'] = (1. - sign) / 2.
        return sign

    def make_step(self, action=None):
        """

        Calculates new state.


        :param action: array of floats, the agent's last action

        :return: observation: array, agent's observation of the environment state


        """
        # take actions
        for i in range(len(action)):
            if action is not None:
                self.sims[i].set_property_values(self.task.action_var, action[i])
            # run simulation
            self.sims[i].run()

    def get_observation(self):
        """
        get state observation from sim.

        :return: NamedTuple, the first state observation of the episode

        """
        blue_obs_list = self.sims[0].get_property_values(self.task.state_var)
        red_obs_list = self.sims[1].get_property_values(self.task.state_var)

        blue_observation = self.normalize_observation(blue_obs_list, red_obs_list)
        red_observation = self.normalize_observation(red_obs_list, blue_obs_list)
        # tuple([np.array([obs]) for obs in obs_list])
        return blue_observation, red_observation

    def process_actions(self, actions, fighter_type, lowpass=True):
        action = actions[fighter_type].astype(np.float)
        action[0] = action[0] * 2. / (act_space['aileron'].n - 1.) - 1.
        action[1] = action[1] * 2. / (act_space['elevator'].n - 1.) - 1.
        action[2] = action[2] * 2. / (act_space['rudder'].n - 1.) - 1.
        action[3] = action[3] * 0.5 / (act_space['throttle'].n - 1.) + 0.4
        return action

    def normalize_observation(self, ego_obs_list, enm_obs_list):
        observation = np.zeros(22)
        init_longitude = self.task.init_position_conditions[c.ic_long_gc_deg]
        init_latitude = self.task.init_position_conditions[c.ic_lat_geod_deg]
        ego_cur_east, ego_cur_north = lonlat2dis(ego_obs_list[0], ego_obs_list[1], init_longitude, init_latitude)
        enm_cur_east, enm_cur_north = lonlat2dis(enm_obs_list[0], enm_obs_list[1], init_longitude, init_latitude)
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

    def get_sim_time(self):
        """ Gets the simulation time from sim, a float. """
        return self.sims[0].get_sim_time()

    def get_state(self):
        pass

    def _get_clipped_state(self):
        pass

    def close(self):
        """Cleans up this environment's objects
        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sims:
            for i in range(len(self.sims)):
                self.sims[i].close()

    def is_terminal(self):
        """

        Checks if the state is terminal.

        :return: bool

        """
        is_not_contained = False                                # not self.observation_space.contains(self.state)
        rets = self.task.is_terminal(self.state, self.sims)
        self.end_reason['blue_fighter_crash'] = rets[0]
        self.end_reason['red_fighter_crash'] = rets[1]
        return is_not_contained or rets[0] or rets[1]

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
        blue_obs_list = self.sims[0].get_property_values(self.task.render_var)
        red_obs_list = self.sims[1].get_property_values(self.task.render_var)
        self.trajectory.append(np.hstack([np.asarray(blue_obs_list), np.asarray(red_obs_list)]))
        return np.hstack([np.asarray(blue_obs_list), np.asarray(red_obs_list)])
