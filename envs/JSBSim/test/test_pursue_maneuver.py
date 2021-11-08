import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.tasks import SingleCombatWithMissileTask
from envs.JSBSim.core.simulatior import MissileSimulator
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir
import torch 
import time
import numpy as np
from typing import Literal


class ManeuverAgent:
    def __init__(self, agent_id: Literal[0, 1], maneuver: Literal['l', 's', 'o']):
        self.model_path = get_root_dir() + '/model/singlecontrol_baseline.pth'
        self.ego_idx = agent_id
        self.restore()
        self.prep_rollout()
        self.step = 0
        self.seconds_per_turn = 7  # hyperparameter
        self.init_heading = None
        if maneuver == 'l':
            self.target_heading_list = [0, 0, 0, 0]
        elif maneuver == 's':
            self.target_heading_list = [np.pi/2, 0, -np.pi/2, -np.pi/2]
        elif maneuver == 'o':
            self.target_heading_list = [np.pi/2, 0, -np.pi/2, -np.pi]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))
        self.step = 0

    def get_action(self, env: SingleCombatEnv, task: SingleCombatWithMissileTask):
        step_list = np.arange(1, 5) * self.seconds_per_turn / env.time_interval
        ego_uid = list(env.jsbsims.keys())[self.ego_idx]
        ego_obs_list = env.jsbsims[ego_uid].get_property_values(task.state_var)
        if self.init_heading is None:
            self.init_heading = ego_obs_list[5]
        delta_heading = 0

        missile = task.check_missile_warning(env, self.ego_idx)
        if missile is None:
            delta_heading = (self.init_heading - ego_obs_list[5])  # ego_obs_list[5] is ego's heading
        elif (np.sum(missile.get_velocity() * env.jsbsims[ego_uid].get_velocity()) >= 0) \
            and (np.linalg.norm(missile.get_position() - env.jsbsims[ego_uid].get_position()) < 4000) \
            or (np.sum(missile.get_velocity() * env.jsbsims[ego_uid].get_velocity()) < 0):
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            target_heading = self.target_heading_list[i]
            delta_heading = (self.init_heading + target_heading - ego_obs_list[5])
            self.step += 1

        observation = np.zeros(8)
        observation[0] = 0                              #  0. ego delta altitude  (unit: 1km)
        observation[1] = in_range_rad(delta_heading)    #  1. ego delta heading   (unit rad)
        observation[2] = ego_obs_list[3]                #  2. ego_roll      (unit: rad)
        observation[3] = ego_obs_list[4]                #  3. ego_pitch     (unit: rad)
        observation[4] = ego_obs_list[6] / 340          #  4. ego_v_north   (unit: mh)
        observation[5] = ego_obs_list[7] / 340          #  5. ego_v_east    (unit: mh)
        observation[6] = ego_obs_list[8] / 340          #  6. ego_v_down    (unit: mh)
        observation[7] = ego_obs_list[9] / 340          #  7. ego_vc        (unit: mh)
        observation = np.expand_dims(observation, axis=0)   # dim: (1,8)
        _action, _, self.rnn_states = self.actor(observation, self.rnn_states, deterministic=True)
        action = _action.detach().cpu().numpy().squeeze()
        return action

    def restore(self):
        self.actor = torch.load(str(self.model_path))

    def prep_rollout(self):
        self.actor.eval()


def test_maneuver():
    env = SingleCombatEnv(config_name='1v1/Missile/test/pursue')
    escape_agent = ManeuverAgent(agent_id=0, maneuver='l')
    pursue_agent = ManeuverAgent(agent_id=1, maneuver='l')
    dt = env.time_interval
    obs = env.reset()
    env.render()
    missile_data = []
    while True:
        actions = [
            escape_agent.get_action(env, env.task),
            pursue_agent.get_action(env, env.task),
        ]
        obs, reward, done, env_info = env.step(actions)
        if "B0101" in env.other_sims.keys():
            sim = env.other_sims["B0101"]
            if isinstance(sim, MissileSimulator):
                missile_data.append((np.linalg.norm(sim.get_velocity()), sim.target_distance))
        env.render()
        if np.array(done).all():
            print(env_info)
            break
    missile_data = np.array(missile_data)
    np.save("missile_data_lvl", missile_data)


if __name__ == "__main__":
    test_maneuver()
