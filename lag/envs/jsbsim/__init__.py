import gymnasium as gym

from lag.envs.jsbsim.envs.singlecontrol_env import SingleControlEnv
from lag.envs.jsbsim.envs.singlecombat_env import SingleCombatEnv
from lag.envs.jsbsim.envs.multiplecombat_env import MultipleCombatEnv


gym.register(
    id="LAG-1P-v0",
    entry_point="lag.envs.jsbsim.envs.singlecontrol_env:SingleControlEnv",
    kwargs={"config_name": "1/heading"},
)

gym.register(
    id="LAG-1P_vs_1B-v0",
    entry_point="lag.envs.jsbsim.envs.singlecombat_env:SingleCombatEnv",
    kwargs={"config_name": "1v1/NoWeapon/vsBaseline"},
)

gym.register(
    id="LAG-1P_vs_1P-v0",
    entry_point="lag.envs.jsbsim.envs.singlecombat_env:SingleCombatEnv",
    kwargs={"config_name": "1v1/NoWeapon/Selfplay"},
)

gym.register(
    id="LAG-1HP_vs_1HP-v0",
    entry_point="lag.envs.jsbsim.envs.singlecombat_env:SingleCombatEnv",
    kwargs={"config_name": "1v1/NoWeapon/HierarchySelfplay"},
)

gym.register(
    id="LAG-1HP2m_vs_1HP2m-v0",
    entry_point="lag.envs.jsbsim.envs.singlecombat_env:SingleCombatEnv",
    kwargs={"config_name": "1v1/ShootMissile/HierarchySelfplay"},
)

gym.register(
    id="LAG-2P_vs_2P-v0",
    entry_point="lag.envs.jsbsim.envs.multiplecombat_env:MultipleCombatEnv",
    kwargs={"config_name": "2v2/NoWeapon/Selfplay"},
)


gym.register(
    id="LAG-2HP_vs_2HP-v0",
    entry_point="lag.envs.jsbsim.envs.multiplecombat_env:MultipleCombatEnv",
    kwargs={"config_name": "2v2/NoWeapon/HierarchySelfplay"},
)
