from .slimevolleygym import SlimeVolleyEnv
import numpy as np


class VolleyballEnv(SlimeVolleyEnv):
    def __init__(self) -> None:
        super().__init__()
        self.from_pixels = False # super setting
        self.atari_mode = True   # super setting
        self.survival_reward = False
        self.num_agents = 2
        self.act_shape = (self.num_agents, 1)
        self.obs_shape = (self.num_agents, *self.observation_space.shape)
        self.done_shape = (self.num_agents, 1)
        self.reward_shape = (self.num_agents, 1)

    def reset(self):
        obs = super().reset()
        return np.array([obs, obs], dtype=np.float32)
    
    def step(self, action: np.ndarray):
        action = action.squeeze()
        _obs, _reward, _done, info = super().step(action[0], action[1])
        obs = np.array([_obs, info["otherObs"]], dtype=np.float32)
        done = np.array([[_done], [_done]], dtype=np.float32)
        reward = np.array([[_reward], [-_reward]], dtype=np.float32)
        if self.survival_reward:
            reward += 0.01
        return obs, reward, done, info