import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatWithMissileTask, SingleCombatWithMissileHierarchicalTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'singlecombat_with_missile':
            self.task = SingleCombatWithMissileTask(self.config)
        elif taskname == 'singlecombat_with_missile_hierarchical':
            self.task = SingleCombatWithMissileHierarchicalTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        self.init_states[0].update({
            'ic_psi_true_deg': self.np_random.uniform(90, 270),
            'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        })
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()
