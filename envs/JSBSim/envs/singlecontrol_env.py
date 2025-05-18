from envs.JSBSim.human_task.HumanFreeFlyTask import HumanFreeFlyTask
from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        elif taskname == "HumanFreeFly":
            self.task = HumanFreeFlyTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')
        
    def load_static_base(self):
        static_base_configs = getattr(self.config, 'static_base_configs', None)
        self.static_bases = {}
        if static_base_configs is not None:
            for uid, base_cfg in static_base_configs.items():
                color = base_cfg.get('color', 'Blue')
                model = base_cfg.get('model', 'BGas Platformse')
                type = base_cfg.get('type', 'Ground+Static+Building')
                init_state = base_cfg.get('init_state', {})
                # Convert geodetic to NEU if needed, or store geodetic directly
                position = self.geodetic_to_neu(
                    init_state.get('ic_long_gc_deg', 120.5),
                    init_state.get('ic_lat_geod_deg', 60.5),
                    init_state.get('ic_h_sl_ft', 1000) * 0.3048  # ft to m
                )
                self.static_bases[uid] = {
                    'color': color,
                    'model': model,
                    'type': type,
                    'position': position,
                    'init_state': init_state
                }
    
    def render_static_base(self, uid, position, color, model, type_, filepath=None):
        """
        Render a static base to the ACMI file.
        Args:
            uid (str): Unique ID for the base
            position (tuple): (north, east, up) in meters (NEU coordinates)
            color (str): Color string
            model (str): Model name
            type_ (str): Type string
        """
        # Convert NEU to geodetic if needed, or use directly
        # Example: lon, lat, alt = NEU2LLA(*position, self.lon0, self.lat0, self.alt0)
        lon, lat, alt = self.neu_to_geodetic(*position) 

        # Compose ACMI line (Tacview format example)
        line = (
            f"{uid},T={lon}|{lat}|{alt}|0|0|0,"
            f"Type={type_},Color={color},Name={model}\n"
        )

        # Write to ACMI file (append mode)
        with open(filepath, "a") as f:
            f.write(line)

    def neu_to_geodetic(self, north, east, up):
        # Implement or import this conversion as needed
        from ..utils.utils import NEU2LLA
        return NEU2LLA(north, east, up, 120.5, 60.5, 0.0)
    

    def geodetic_to_neu(self, lon, lat, alt):
        # Implement or import this conversion as needed
        from ..utils.utils import LLA2NEU
        return LLA2NEU(lon, lat, alt, 120.5, 60.5, 0.0)


    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        # -> initial longitude and latitude: in heading.yaml ic_lat_geod_deg 60; ic_long_gc_deg 120
        # -> current position can be obtained by sim.get_property_values([c.position_lat_geod_deg, c.position_long_gc_deg])
        # -> target position can be obtained by sim.get_property_values([c.target_latitude_geod_deg, c.target_longitude_geod_deg])
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()
