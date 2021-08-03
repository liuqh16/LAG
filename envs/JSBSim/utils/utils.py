import os
import math
import yaml
import numpy as np


def parse_config(filename):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def lonlat2dis(lon, lat, init_lon, init_lat):
    """Convert longitude&latitude into xy distance

    Args:
        lon (float): lontitude of current point
        lat (float): latitude of current point
        init_lon (float): lontitude of original point
        init_lat (float): latitude of original point

    Returns:
        (np.array): (east, north), unit: m
    """
    east = (np.deg2rad(lon) - np.deg2rad(init_lon)) * np.cos(np.deg2rad(init_lat)) * 6371 * 1.734
    north = (np.deg2rad(lat) - np.deg2rad(init_lat)) * 6371 * 11.1319 / 11.11949266
    return np.array([east, north]) * 1000


def get_AO_TA_R(ego_feature, enemy_feature):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd), unit: km, mh

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enemy_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))

    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))
    return ego_AO, ego_TA, R


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def _in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def unit_converse(x, left: str, right: str):
    if left == 'ft' and right == 'm':
        return x * 0.3048
    elif left == 'm' and right == 'ft':
        return x * 3.2808
    else:
        raise NotImplementedError


def shortest_ac_dist(x, y, x1, y1, x2, y2):
    """
    Compute the shortest distance (in m) between aircraft coord (lat, lon) and the line between two points (lat1,lon1) and (lat2, lon2)
    >>> round(shortest_ac_dist(40.759809, -73.976264, 40.758492, -73.975105, 40.759752, -73.974215))
    160
    >>> round(shortest_ac_dist(40.759168, -73.976741, 40.758492, -73.975105, 40.759752, -73.974215))
    160
    """
    # print(x, y, x1, y1, x2, y2)
    # equation of line (lat1,lon1) -> (lat2, lon2): y = s*x + m
    # slope
    s = (y2 - y1) / (x2 - x1 + 0.00000000001)
    # find m: m = y - s*x
    m = y1 - s * x1
    # coeff of line equation ay + bx + c = 0
    a = -1
    b = s
    c = m
    # compute shortest distance between aircarft and the line = abs(a*x0 + b*y0 + c) / sqrt(a²+b²))
    s_ac_dist = math.fabs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)
    # print("s_ac_dist", s_ac_dist)
    return s_ac_dist
