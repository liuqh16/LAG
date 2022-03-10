from gym.envs.registration import register
from .slimevolley_wrapper import VolleyballEnv

register(
    id='Volleyball-v0',
    entry_point='envs.slimevolley:VolleyballEnv'
)