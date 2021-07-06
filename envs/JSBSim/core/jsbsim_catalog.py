import math
from enum import Enum
from .catalog import Property


class JsbsimCatalog(Property, Enum):
    """

    A class to store and customize jsbsim properties

    """

    # position and attitude

    position_h_sl_ft = Property("position/h-sl-ft", "altitude above mean sea level [ft]", -1400, 85000)
    position_h_agl_ft = Property(
        "position/h-agl-ft", "altitude above ground level [ft]", position_h_sl_ft.min, position_h_sl_ft.max
    )
    attitude_pitch_rad = Property("attitude/pitch-rad", "pitch [rad]", -0.5 * math.pi, 0.5 * math.pi, access="R")
    attitude_theta_rad = Property("attitude/theta-rad", "rad", access="R")
    attitude_theta_deg = Property("attitude/theta-deg", "deg", access="R")
    attitude_roll_rad = Property("attitude/roll-rad", "roll [rad]", -math.pi, math.pi, access="R")
    attitude_phi_rad = Property("attitude/phi-rad", "rad", access="R")
    attitude_phi_deg = Property("attitude/phi-deg", "deg", access="R")
    attitude_heading_true_rad = Property("attitude/heading-true-rad", "rad", access="R")
    attitude_psi_deg = Property("attitude/psi-deg", "heading [deg]", 0, 360, access="R")
    attitude_psi_rad = Property("attitude/psi-rad", "rad", access="R")
    aero_beta_deg = Property("aero/beta-deg", "sideslip [deg]", -180, +180, access="R")
    position_lat_geod_deg = Property("position/lat-geod-deg", "geocentric latitude [deg]", -90, 90, access="R")
    position_lat_geod_rad = Property("position/lat-geod-rad", "rad", access="R")
    position_lat_gc_deg = Property("position/lat-gc-deg", "deg")
    position_lat_gc_rad = Property("position/lat-gc-rad", "rad")
    position_long_gc_deg = Property("position/long-gc-deg", "geodesic longitude [deg]", -180, 180)
    position_long_gc_rad = Property("position/long-gc-rad", "rad")
    position_distance_from_start_mag_mt = Property(
        "position/distance-from-start-mag-mt", "distance travelled from starting position [m]", access="R"
    )
    position_distance_from_start_lat_mt = Property("position/distance-from-start-lat-mt", "mt", access="R")
    position_distance_from_start_lon_mt = Property("position/distance-from-start-lon-mt", "mt", access="R")
    position_epa_rad = Property("position/epa-rad", "rad", access="R")
    position_geod_alt_ft = Property("position/geod-alt-ft", "ft", access="R")
    position_radius_to_vehicle_ft = Property("position/radius-to-vehicle-ft", "ft", access="R")
    position_terrain_elevation_asl_ft = Property("position/terrain-elevation-asl-ft", "ft")

    # velocities

    velocities_u_fps = Property("velocities/u-fps", "body frame x-axis velocity [ft/s]", -2200, 2200, access="R")
    velocities_v_fps = Property("velocities/v-fps", "body frame y-axis velocity [ft/s]", -2200, 2200, access="R")
    velocities_w_fps = Property("velocities/w-fps", "body frame z-axis velocity [ft/s]", -2200, 2200, access="R")
    velocities_v_north_fps = Property("velocities/v-north-fps", "velocity true north [ft/s]", -2200, 2200, access="R")
    velocities_v_east_fps = Property("velocities/v-east-fps", "velocity east [ft/s]", -2200, 2200, access="R")
    velocities_v_down_fps = Property("velocities/v-down-fps", "velocity downwards [ft/s]", -2200, 2200, access="R")
    velocities_vc_fps = Property("velocities/vc-fps", "airspeed in knots", 0, 4400, access="R")
    velocities_h_dot_fps = Property("velocities/h-dot-fps", "rate of altitude change [ft/s]", access="R")
    velocities_u_aero_fps = Property("velocities/u-aero-fps", "fps", access="R")
    velocities_v_aero_fps = Property("velocities/v-aero-fps", "fps", access="R")
    velocities_w_aero_fps = Property("velocities/w-aero-fps", "fps", access="R")
    velocities_mach = Property("velocities/mach", "", access="R")
    velocities_machU = Property("velocities/machU", "", access="R")
    velocities_eci_velocity_mag_fps = Property("velocities/eci-velocity-mag-fps", "fps", access="R")
    velocities_vc_kts = Property("velocities/vc-kts", "kts", access="R")
    velocities_ve_fps = Property("velocities/ve-fps", "fps", access="R")
    velocities_ve_kts = Property("velocities/ve-kts", "kts", access="R")
    velocities_vg_fps = Property("velocities/vg-fps", "fps", access="R")
    velocities_vt_fps = Property("velocities/vt-fps", "fps", access="R")
    velocities_p_rad_sec = Property("velocities/p-rad_sec", "roll rate [rad/s]", -2 * math.pi, 2 * math.pi, access="R")
    velocities_q_rad_sec = Property(
        "velocities/q-rad_sec", "pitch rate [rad/s]", -2 * math.pi, 2 * math.pi, access="R"
    )
    velocities_r_rad_sec = Property("velocities/r-rad_sec", "yaw rate [rad/s]", -2 * math.pi, 2 * math.pi, access="R")
    velocities_p_aero_rad_sec = Property("velocities/p-aero-rad_sec", "rad/sec", access="R")
    velocities_q_aero_rad_sec = Property("velocities/q-aero-rad_sec", "rad/sec", access="R")
    velocities_r_aero_rad_sec = Property("velocities/r-aero-rad_sec", "rad/sec", access="R")
    velocities_phidot_rad_sec = Property("velocities/phidot-rad_sec", "rad/s", -2 * math.pi, 2 * math.pi, access="R")
    velocities_thetadot_rad_sec = Property(
        "velocities/thetadot-rad_sec", "rad/s", -2 * math.pi, 2 * math.pi, access="R"
    )
    velocities_psidot_rad_sec = Property("velocities/psidot-rad_sec", "rad/sec", -2 * math.pi, 2 * math.pi, access="R")

    # Acceleration

    accelerations_pdot_rad_sec2 = Property(
        "accelerations/pdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, access="R"
    )
    accelerations_qdot_rad_sec2 = Property(
        "accelerations/qdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, access="R"
    )
    accelerations_rdot_rad_sec2 = Property(
        "accelerations/rdot-rad_sec2", "rad/sÂ²", -(8 / 180) * math.pi, (8 / 180) * math.pi, access="R"
    )
    accelerations_vdot_ft_sec2 = Property("accelerations/vdot-ft_sec2", "ft/sÂ²", -4.0, 4.0, access="R")
    accelerations_wdot_ft_sec2 = Property("accelerations/wdot-ft_sec2", "ft/sÂ²", -4.0, 4.0, access="R")
    accelerations_udot_ft_sec2 = Property("accelerations/udot-ft_sec2", "ft/sÂ²", -4.0, 4.0, access="R")
    accelerations_a_pilot_x_ft_sec2 = Property(
        "accelerations/a-pilot-x-ft_sec2", "pilot body x-axis acceleration [ft/sÂ²]", access="R"
    )
    accelerations_a_pilot_y_ft_sec2 = Property(
        "accelerations/a-pilot-y-ft_sec2", "pilot body y-axis acceleration [ft/sÂ²]", access="R"
    )
    accelerations_a_pilot_z_ft_sec2 = Property(
        "accelerations/a-pilot-z-ft_sec2", "pilot body z-axis acceleration [ft/sÂ²]", access="R"
    )
    accelerations_n_pilot_x_norm = Property(
        "accelerations/n-pilot-x-norm", "pilot body x-axis acceleration, normalised", access="R"
    )
    accelerations_n_pilot_y_norm = Property(
        "accelerations/n-pilot-y-norm", "pilot body y-axis acceleration, normalised", access="R"
    )
    accelerations_n_pilot_z_norm = Property(
        "accelerations/n-pilot-z-norm", "pilot body z-axis acceleration, normalised", access="R"
    )

    # aero

    aero_alpha_deg = Property("aero/alpha-deg", "deg", access="R")
    aero_beta_rad = Property("aero/beta-rad", "rad", access="R")

    # controls state

    @staticmethod
    def update_equal_engine_props(sim, prop):
        """
        Update the given property for all engines
        :param sim: simulation to use
        :param prop: property to update
        """
        value = sim.get_property_value(prop)
        n = sim.jsbsim_exec.get_propulsion().get_num_engines()
        for i in range(1, n):
            sim.jsbsim_exec.set_property_value(prop.name_jsbsim + "[" + str(i) + "]", value)

    def update_equal_throttle_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_throttle_pos_norm)

    def update_equal_mixture_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_mixture_pos_norm)

    def update_equal_feather_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_feather_pos_norm)

    def update_equal_advance_pos(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_advance_pos_norm)

    fcs_left_aileron_pos_norm = Property("fcs/left-aileron-pos-norm", "left aileron position, normalised", -1, 1)
    fcs_right_aileron_pos_norm = Property("fcs/right-aileron-pos-norm", "right aileron position, normalised", -1, 1)
    fcs_elevator_pos_norm = Property("fcs/elevator-pos-norm", "elevator position, normalised", -1, 1)
    fcs_rudder_pos_norm = Property("fcs/rudder-pos-norm", "rudder position, normalised", -1, 1)
    fcs_flap_pos_norm = Property("fcs/flap-pos-norm", "flap position, normalised", 0, 1)
    fcs_speedbrake_pos_norm = Property("fcs/speedbrake-pos-norm", "speedbrake position, normalised", 0, 1)
    fcs_spoiler_pos_norm = Property("fcs/spoiler-pos-norm", "normalised")
    fcs_steer_pos_deg = Property("fcs/steer-pos-deg", "deg")
    fcs_throttle_pos_norm = Property(
        "fcs/throttle-pos-norm", "throttle position, normalised", 0, 1, update=update_equal_throttle_pos
    )
    fcs_mixture_pos_norm = Property("fcs/mixture-pos-norm", "normalised", update=update_equal_mixture_pos)
    gear_gear_pos_norm = Property("gear/gear-pos-norm", "landing gear position, normalised", 0, 1)
    gear_num_units = Property("gear/num-units", "number of gears", access="R")
    fcs_feather_pos_norm = Property("fcs/feather-pos-norm", "normalised", update=update_equal_feather_pos)
    fcs_advance_pos_norm = Property("fcs/advance-pos-norm", "normalised", update=update_equal_advance_pos)

    # controls command

    def update_equal_throttle_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_throttle_cmd_norm)

    def update_equal_mixture_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_mixture_cmd_norm)

    def update_equal_advance_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_advance_cmd_norm)

    def update_equal_feather_cmd(sim):
        JsbsimCatalog.update_equal_engine_props(sim, JsbsimCatalog.fcs_feather_cmd_norm)

    @staticmethod
    def update_equal_brake_props(sim):
        value = sim.get_property_value(JsbsimCatalog.fcs_center_brake_cmd_norm)
        sim.jsbsim_exec.set_property_value(JsbsimCatalog.fcs_left_brake_cmd_norm.name_jsbsim, value)
        sim.jsbsim_exec.set_property_value(JsbsimCatalog.fcs_right_brake_cmd_norm.name_jsbsim, value)

    def update_equal_brake_cmd(sim):
        JsbsimCatalog.update_equal_brake_props(sim)

    fcs_aileron_cmd_norm = Property("fcs/aileron-cmd-norm", "aileron commanded position, normalised", -1.0, 1.0)
    fcs_elevator_cmd_norm = Property("fcs/elevator-cmd-norm", "elevator commanded position, normalised", -1.0, 1.0)
    fcs_rudder_cmd_norm = Property("fcs/rudder-cmd-norm", "rudder commanded position, normalised", -1.0, 1.0)
    fcs_throttle_cmd_norm = Property(
        "fcs/throttle-cmd-norm", "throttle commanded position, normalised", 0.0, 0.9, update=update_equal_throttle_cmd
    )
    fcs_mixture_cmd_norm = Property(
        "fcs/mixture-cmd-norm", "engine mixture setting, normalised", 0.0, 1.0, update=update_equal_mixture_cmd
    )
    gear_gear_cmd_norm = Property("gear/gear-cmd-norm", "all landing gear commanded position, normalised", 0.0, 1.0)
    fcs_speedbrake_cmd_norm = Property("fcs/speedbrake-cmd-norm", "normalised")
    fcs_left_brake_cmd_norm = Property("fcs/left-brake-cmd-norm", "Left brake command(normalized)", 0.0, 1.0)
    fcs_center_brake_cmd_norm = Property(
        "fcs/center-brake-cmd-norm", "normalised", 0.0, 1.0, update=update_equal_brake_cmd
    )
    fcs_right_brake_cmd_norm = Property("fcs/right-brake-cmd-norm", "Right brake command(normalized)", 0.0, 1.0)
    fcs_spoiler_cmd_norm = Property("fcs/spoiler-cmd-norm", "normalised")
    fcs_flap_cmd_norm = Property("fcs/flap-cmd-norm", "normalised")
    fcs_steer_cmd_norm = Property("fcs/steer-cmd-norm", "Steer command(normalized)", -1.0, 1.0)
    fcs_advance_cmd_norm = Property("fcs/advance-cmd-norm", "normalised", update=update_equal_advance_cmd)
    fcs_feather_cmd_norm = Property("fcs/feather-cmd-norm", "normalised", update=update_equal_feather_cmd)

    # initial conditions

    ic_h_sl_ft = Property("ic/h-sl-ft", "initial altitude MSL [ft]", position_h_sl_ft.min, position_h_sl_ft.max)
    ic_h_agl_ft = Property("ic/h-agl-ft", "", position_h_sl_ft.min, position_h_sl_ft.max)
    ic_geod_alt_ft = Property("ic/geod-alt-ft", "ft")
    ic_sea_level_radius_ft = Property("ic/sea-level-radius-ft", "ft")
    ic_terrain_elevation_ft = Property("ic/terrain-elevation-ft", "ft")
    ic_long_gc_deg = Property("ic/long-gc-deg", "initial geocentric longitude [deg]")
    ic_long_gc_rad = Property("ic/long-gc-rad", "rad")
    ic_lat_gc_deg = Property("ic/lat-gc-deg", "deg")
    ic_lat_gc_rad = Property("ic/lat-gc-rad", "rad")
    ic_lat_geod_deg = Property("ic/lat-geod-deg", "initial geodesic latitude [deg]")
    ic_lat_geod_rad = Property("ic/lat-geod-rad", "rad")
    ic_psi_true_deg = Property(
        "ic/psi-true-deg", "initial (true) heading [deg]", attitude_psi_deg.min, attitude_psi_deg.max
    )
    ic_psi_true_rad = Property("ic/psi-true-rad", "rad")
    ic_theta_deg = Property("ic/theta-deg", "deg")
    ic_theta_rad = Property("ic/theta-rad", "rad")
    ic_phi_deg = Property("ic/phi-deg", "deg")
    ic_phi_rad = Property("ic/phi-rad", "rad")
    ic_alpha_deg = Property("ic/alpha-deg", "deg")
    ic_alpha_rad = Property("ic/alpha-rad", "rad")
    ic_beta_deg = Property("ic/beta-deg", "deg")
    ic_beta_rad = Property("ic/beta-rad", "rad")
    ic_gamma_deg = Property("ic/gamma-deg", "deg")
    ic_gamma_rad = Property("ic/gamma-rad", "rad")
    ic_mach = Property("ic/mach", "")
    ic_u_fps = Property("ic/u-fps", "body frame x-axis velocity; positive forward [ft/s]")
    ic_v_fps = Property("ic/v-fps", "body frame y-axis velocity; positive right [ft/s]")
    ic_w_fps = Property("ic/w-fps", "body frame z-axis velocity; positive down [ft/s]")
    ic_p_rad_sec = Property("ic/p-rad_sec", "roll rate [rad/s]")
    ic_q_rad_sec = Property("ic/q-rad_sec", "pitch rate [rad/s]")
    ic_r_rad_sec = Property("ic/r-rad_sec", "yaw rate [rad/s]")
    ic_roc_fpm = Property("ic/roc-fpm", "initial rate of climb [ft/min]")
    ic_roc_fps = Property("ic/roc-fps", "fps")
    ic_vc_kts = Property("ic/vc-kts", "kts")
    ic_vd_fps = Property("ic/vd-fps", "fps")
    ic_ve_fps = Property("ic/ve-fps", "fps")
    ic_ve_kts = Property("ic/ve-kts", "kts")
    ic_vg_fps = Property("ic/vg-fps", "fps")
    ic_vg_kts = Property("ic/vg-kts", "kts")
    ic_vn_fps = Property("ic/vn-fps", "fps")
    ic_vt_fps = Property("ic/vt-fps", "fps")
    ic_vt_kts = Property("ic/vt-kts", "kts")
    ic_vw_bx_fps = Property("ic/vw-bx-fps", "fps")
    ic_vw_by_fps = Property("ic/vw-by-fps", "fps")
    ic_vw_bz_fps = Property("ic/vw-bz-fps", "fps")
    ic_vw_dir_deg = Property("ic/vw-dir-deg", "deg")
    ic_vw_down_fps = Property("ic/vw-down-fps", "fps")
    ic_vw_east_fps = Property("ic/vw-east-fps", "fps")
    ic_vw_mag_fps = Property("ic/vw-mag-fps", "fps")
    ic_vw_north_fps = Property("ic/vw-north-fps", "fps")
    ic_targetNlf = Property("ic/targetNlf", "")

    # engines

    propulsion_engine_set_running = Property("propulsion/engine/set-running", "engine running (0/1 bool)")
    propulsion_set_running = Property("propulsion/set-running", "set engine running (-1 for all engines)", access="W")
    propulsion_tank_contents_lbs = Property("propulsion/tank/contents-lbs", "")

    # simulation

    simulation_dt = Property("simulation/dt", "JSBSim simulation timestep [s]", access="R")
    simulation_sim_time_sec = Property("simulation/sim-time-sec", "Simulation time [s]", access="R")
    simulation_do_simple_trim = Property("simulation/do_simple_trim", "", access="W")

    # Auto Pilot
    ap_vg_hold = Property("ap/vg-hold", "Auto Pilot ON OFF")
