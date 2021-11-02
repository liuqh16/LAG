from collections import namedtuple
import numpy as np
from os import path
import jsbsim
from .catalog import Property, Catalog
from ..utils.utils import get_root_dir, LLA2NEU


class Simulation:
    """A class which wraps an instance of JSBSim and manages communication with it.
    """

    def __init__(self, aircraft_name="f15", init_conditions=None, origin_point=(120.0, 60.0, 0.0), jsbsim_freq=60, agent_interaction_steps=5):
        """Constructor. Creates an instance of JSBSim, loads an aircraft and sets initial conditions.

        Args:
            aircraft_name (str): name of aircraft to be loaded. Default = `"f15"`.
                JSBSim looks for file './data/aircraft_name/aircraft_name.xml' from root dir.
            init_conditions (dict): dict mapping properties to their initial values. Default = `None`, causing a default set of initial props to be used.
            origin_point (tuple): origin point (longitude, latitude, altitude) of the global Combat Field. Default = `(120.0, 60.0, 0.0)`
            jsbsim_freq (int): JSBSim integration frequency. Default = `60`.
            agent_interaction_steps (int): simulation steps before the agent interact. Default = `5`.
        """
        self.jsbsim_exec = jsbsim.FGFDMExec(path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)  # requests JSBSim not to output any messages whatsoever
        self.jsbsim_exec.load_model(aircraft_name)
        # collect all jsbsim properties in Catalog (use Catalog.pop to remove useless props)
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))
        dt = 1 / jsbsim_freq
        self.lon0, self.lat0, self.alt0 = origin_point
        self.jsbsim_exec.set_dt(dt)
        self.agent_interaction_steps = agent_interaction_steps
        self.initialise(init_conditions)

        # properties
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._poseture = np.zeros(3)
        self._velocity = np.zeros(3)
        self._update_properties()

    def initialise(self, init_conditions):
        self.set_initial_conditions(init_conditions)
        success = self.jsbsim_exec.run_ic()
        self.propulsion_init_running(-1)
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

    def propulsion_init_running(self, i):
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        if i >= 0:
            if i >= n:
                raise IndexError("Tried to initialize a non-existent engine!")
            propulsion.get_engine(i).init_running()
            propulsion.get_steady_state()
        else:
            for j in range(n):
                propulsion.get_engine(j).init_running()
            propulsion.get_steady_state()

    def set_initial_conditions(self, init_conditions=None):
        """

        Loads init_conditions values in JSBSim.

        :param init_conditions: dict mapping properties to their initial values

        """

        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self.set_property_value(prop, value)

    def run(self):
        """

        Runs JSBSim simulation until the agent interacts and update custom properties.


        JSBSim monitors the simulation and detects whether it thinks it should

        end, e.g. because a simulation time was specified. False is returned

        if JSBSim termination criteria are met.



        :return: bool, False if sim has met JSBSim termination criteria else True.

        """
        for _ in range(self.agent_interaction_steps):
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed.")
        self._update_properties()

        return result

    def _update_properties(self):
        # update position
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        # update poseture
        self._poseture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad,
        ])
        # update velocity
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps,
        ])

    def get_geodetic(self):
        """(lontitude, latitude, altitude), unit: °, m"""
        return self._geodetic

    def get_position(self):
        """(north, east, down), unit: m"""
        return self._position

    def get_rpy(self):
        """(roll, pitch, yaw), unit: rad"""
        return self._poseture

    def get_velocity(self):
        """(v_north, v_east, v_down), unit: m/s"""
        return self._velocity

    def get_sim_time(self):
        """ Gets the simulation time from JSBSim, a float. """
        return self.jsbsim_exec.get_sim_time()

    def close(self):
        """ Closes the simulation and any plots. """

        if self.jsbsim_exec:
            self.jsbsim_exec = None

    def get_property_values(self, props):
        """

        Get the values of the specified properties

        :param props: list of Properties

        : return: NamedTuple with properties name and their values

        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """

        Set the values of the specified properties

        :param props: list of Properties

        :param values: list of float

        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """
        Get the value of the specified property from the JSBSim simulation

        :param prop: Property

        :return : float
        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """
        Set the values of the specified property

        :param prop: Property

        :param value: float

        """
        # set value in property bounds
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def get_sim_state(self):
        return {prop: self.get_property_value(prop) for prop in Catalog.values()}
