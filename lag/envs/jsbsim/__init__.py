import gymnasium as gym

from lag.envs.jsbsim.envs.singlecontrol_env import SingleControlEnv
from lag.envs.jsbsim.envs.singlecombat_env import SingleCombatEnv
from lag.envs.jsbsim.envs.multiplecombat_env import MultipleCombatEnv


gym.register(
    id="LAG-SingleControl-Heading-v0",
    entry_point="lag.envs.jsbsim.envs.singlecontrol_env:SingleControlEnv",
    kwargs={"config_name": "1/heading"},
)
