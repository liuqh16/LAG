from .env_base import BaseEnv
from ..tasks.singlecombat_task import SingleCombatTask
from ..tasks.singlecombat_with_missle_task import SingleCombatWithMissileTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agent_ids) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':
            self.task = SingleCombatTask(self.config)
        elif taskname == 'singlecombat_with_missile':
            self.task = SingleCombatWithMissileTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        return self.get_obs()

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state for sim in self._jsbsims.values()]
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self._jsbsims.values()):
            sim.reload(init_states[idx])
        self._tempsims.clear()
