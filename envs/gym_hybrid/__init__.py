from gym.envs.registration import register
from .environments import MovingEnv
from .environments import SlidingEnv

register(
    id='Moving-v0',
    entry_point='envs.gym_hybrid:MovingEnv',
)
register(
    id='Sliding-v0',
    entry_point='envs.gym_hybrid:SlidingEnv',
)