import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from math import sin, cos, asin


class MissileConfig(object):
    def __init__(self):
        self.K = 6
        self.dt = 1 / 12.
        self.missile_vel = 600
        self.max_missile_acc = 200
        self.max_distance = 6000

        self.shoot_max_distance = 6000    # [m]
        self.shoot_max_angle = 60         # [deg]
        self.shoot_lock_time = 1          # [s]

        self.hit_distance = 100           # [m]
        self.missile_last_time = 15       # [s]


class MissileCore(object):
    def __init__(self):
        self.args = MissileConfig()
        self.num_step = 0
        self.missile_state, self.seeker_state = None, None
        self.shoot_list = deque(maxlen=int(self.args.shoot_lock_time / self.args.dt))

    def missile_accel(self, missile_state, commanded_accel):
        Vm = missile_state[3]
        psi_m = missile_state[4]
        gma_m = missile_state[5]

        Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                          [0, 1, 0],
                          [sin(gma_m), 0, cos(gma_m)]]).dot(
            np.array([[cos(psi_m), sin(psi_m), 0],
                      [-sin(psi_m), cos(psi_m), 0],
                      [0, 0, 1]]))

        Vm_inert = Rie_m.T.dot(np.array([Vm, 0, 0]).T)

        ac_body_fixed = Rie_m.dot(commanded_accel.T)

        ax = ac_body_fixed[0]
        ay = ac_body_fixed[1]
        az = ac_body_fixed[2]

        return np.array([Vm_inert[0], Vm_inert[1], Vm_inert[2],
                         ax, ay / (Vm * cos(gma_m)), -az / Vm])

    def _vect_cross(self, a, b):
        ax, ay, az = a
        bx, by, bz = b
        return np.array([ay*bz-az*by, az*bx-ax*bz, ax*by - ay*bx])

    def constraint_acc(self, raw_acc_vector):
        a_value = np.linalg.norm(raw_acc_vector)
        if a_value > self.args.max_missile_acc:
            raw_acc_vector = raw_acc_vector * self.args.max_missile_acc / a_value
        return raw_acc_vector

    def pn_guidance(self, seeker_state):
        """
        provides commanded accelerations using proportional navigation law.
        """
        Rx = seeker_state[0]
        Ry = seeker_state[1]
        Rz = seeker_state[2]
        Vx = seeker_state[3]
        Vy = seeker_state[4]
        Vz = seeker_state[5]
        R = np.linalg.norm([Rx, Ry, Rz])
        omega = self._vect_cross(np.array([Vx, Vy, Vz]), np.array([Rx, Ry, Rz])) / (R ** 2)
        adot = self._vect_cross(np.array([Vx, Vy, Vz]), omega) / np.linalg.norm(seeker_state[3:])
        Vc = (Rx * Vx + Ry * Vy + Rz * Vz) / R
        ach = adot * self.args.K * Vc
        return self.constraint_acc(ach)

    def step(self, target_state):
        """
        target_state:   [x, y, z, V, psi--heading, gma-climbing]
        """
        self.num_step += 1
        commanded_accel = self.pn_guidance(self.seeker_state.copy())
        self.missile_state = self._rK4_step(self.missile_state.copy(), self.missile_accel,
                                            self.args.dt, params=commanded_accel)
        self.seeker_state = self.get_seeker_state(target_state, self.missile_state)
        return self.missile_state.copy()

    def get_seeker_state(self, target_state, missile_state):
        """
        1) compute the velocity of both the target and missile in inertial coordination.
        2) compute the relative position and velocity in inner coordination.
        """
        # Target position
        RTx = target_state[0]
        RTy = target_state[1]
        RTz = target_state[2]

        # Target velocity and heading
        VT = target_state[3]
        psi_t = target_state[4]     # heading
        gma_t = target_state[5]     # climbing

        # Missile Position
        RMx = missile_state[0]
        RMy = missile_state[1]
        RMz = missile_state[2]

        # Missile velocity and heading
        VM = missile_state[3]
        psi_m = missile_state[4]
        gma_m = missile_state[5]

        Rie_t = np.array([[cos(gma_t), 0, -sin(gma_t)],
                          [0, 1, 0],
                          [sin(gma_t), 0, cos(gma_t)]]).dot(
            np.array([[cos(psi_t), sin(psi_t), 0],
                      [-sin(psi_t), cos(psi_t), 0],
                      [0, 0, 1]]))

        Rie_m = np.array([[cos(gma_m), 0, -sin(gma_m)],
                          [0, 1, 0],
                          [sin(gma_m), 0, cos(gma_m)]]).dot(
            np.array([[cos(psi_m), sin(psi_m), 0],
                      [-sin(psi_m), cos(psi_m), 0],
                      [0, 0, 1]]))

        # get relative velocity in inertial coordinates
        VT_inert = Rie_t.T.dot(np.array([VT, 0, 0]).T)
        VM_inert = Rie_m.T.dot(np.array([VM, 0, 0]).T)

        return np.array([RTx - RMx, RTy - RMy, RTz - RMz,
                         VT_inert[0] - VM_inert[0], VT_inert[1] - VM_inert[1], VT_inert[2] - VM_inert[2]])

    def _rK4_step(self, state, derivs_fn, dt, params=None, time_invariant=True, t=0):
        # One step of time-invariant RK4 integration
        if params is not None:
            k1 = dt * derivs_fn(state, params)
            k2 = dt * derivs_fn(state + k1 / 2, params)
            k3 = dt * derivs_fn(state + k2 / 2, params)
            k4 = dt * derivs_fn(state + k3, params)

            return state + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            k1 = dt * derivs_fn(state)
            k2 = dt * derivs_fn(state + k1 / 2)
            k3 = dt * derivs_fn(state + k2 / 2)
            k4 = dt * derivs_fn(state + k3)

            return state + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Missile3D(MissileCore):
    def __init__(self, rule_shoot=True):
        super(Missile3D, self).__init__()
        self.max_time = 20
        self.rule_shoot = rule_shoot
        self.missile_pos_vel = None
        self.flag_missile_shot = False
        self.flag_missile_alive = False
        self.flag_missile_during_period = [False, False]

        self.pre_distance = self.args.shoot_max_distance
        self.incre_distance = []

    def _reset(self, initial_missile, initial_target):
        """
        initial_missile:    [x_ego, y_ego, z_ego, vx_ego, vy_ego, vz_ego, Vmissile]
        initial_target:     [x_enm, y_enm, z_enm, vx_enm, vy_enm, vz_enm]
        """
        # 1) the target's states
        target_pos = initial_target[:3]
        target_V = np.linalg.norm(initial_target[3:6])
        psi, gma = self._transfer2angles(initial_target)
        target_state = np.array([*target_pos, target_V, psi, gma])

        # 2) the missile's states
        missile_pos = initial_missile[:3]
        missile_V = initial_missile[6]
        psi_m_0, gma_m_0 = self._transfer2angles(initial_missile)
        missile_state = np.array([*missile_pos, missile_V, psi_m_0, gma_m_0])

        # 3) the seeker's states
        seeker_state = self.get_seeker_state(target_state, missile_state)
        self.missile_state = missile_state.copy()
        self.seeker_state = seeker_state.copy()
        return missile_state

    def missile_step(self, ego_fighter_state, enm_fighter_state):
        # flag_consider_missile = False
        info = {'launched': False, 'mask_enm': False, 'hit': False, 'crash': False}
        flag_shoot = self.determine_shoot(ego_fighter_state, enm_fighter_state)
        if flag_shoot and self.flag_missile_shot is False and self.rule_shoot:
            print('******************************************************************')
            print('The blue fighter has launched the missile')
            print('******************************************************************')
            self.flag_missile_shot = True
            self.flag_missile_alive = True
            self.flag_missile_during_period.append(True)

        if self.flag_missile_shot is False:
            return None, info  # Do not mask the enemy fighter's information.
        # the missile has been launched.
        # 1) reset
        if self.seeker_state is None:
            initial_missile = np.array([*ego_fighter_state, self.args.missile_vel])
            initial_target = enm_fighter_state
            missile_state = self._reset(initial_missile, initial_target)
            self.missile_pos_vel = self._transfer2raw(missile_state)
        else:
            target_pos = enm_fighter_state[:3]
            target_V = np.linalg.norm(enm_fighter_state[3:])
            psi, gma = self._transfer2angles(enm_fighter_state)
            target_state = np.array([*target_pos, target_V, psi, gma])
            missile_state = self.step(target_state)
            self.missile_pos_vel = self._transfer2raw(missile_state)
        info['launched'] = self.flag_missile_shot
        info['hit'] = self.determine_hit(self.missile_pos_vel, enm_fighter_state)
        info['crash'] = self.determine_missile_crash()
        info['mask_enm'] = False if info['hit'] or info['crash'] else True
        self.flag_missile_alive = info['mask_enm']
        self.flag_missile_during_period.append(info['hit'] or info['crash'])
        if not self.flag_missile_alive:
            self.missile_pos_vel = None
        return self.missile_pos_vel, info   # Mask the enemy fighter's information.

    @property
    def switch_situation(self):
        return np.sum(np.asarray(self.flag_missile_during_period[-2:])) % 2

    def determine_shoot(self, ego_state, enm_state):
        distance = np.linalg.norm(ego_state[:3] - enm_state[:3])
        pos_vector = enm_state[:3] - ego_state[:3]
        vel_vector = ego_state[3:]
        ego_angle = np.arccos(np.sum(pos_vector * vel_vector) / (distance * np.linalg.norm(vel_vector)))
        ego_angle = np.rad2deg(ego_angle)
        shoot = 0
        if ego_angle < self.args.shoot_max_angle:
            shoot = 1
        self.shoot_list.append(shoot)
        shoot_condition = np.sum(self.shoot_list) >= self.shoot_list.maxlen and distance <= self.args.shoot_max_distance
        flag_shoot = shoot_condition
        return flag_shoot

    def determine_missile_crash(self, ):
        cur_time_step = self.num_step * self.args.dt
        flag_missile_crash = False
        # print(self.num_step, self.incre_distance[-12:], np.sum(self.incre_distance[-12:]))
        if cur_time_step >= self.args.missile_last_time or np.sum(self.incre_distance[-12:]) >= 12:
            flag_missile_crash = True
        return flag_missile_crash

    def determine_hit(self, ego_missile_state, enm_fighter_state):
        distance = np.linalg.norm(ego_missile_state[:3] - enm_fighter_state[:3])
        self.incre_distance.append(np.sign(distance - self.pre_distance))
        self.pre_distance = distance
        flag_hit = False
        if distance <= self.args.hit_distance:
            flag_hit = True
        return flag_hit

    def _transfer2angles(self, pos_vel):
        vx, vy, vz = pos_vel[3:6]
        R = np.linalg.norm([vx, vy, vz])

        psi_rec = np.arctan2(vy, vx)  # yaw    ---- heading angle
        gma_rec = -asin(vz / R)       # pitch  ---- climb angle
        return [psi_rec, gma_rec]

    def _transfer2raw(self, target_state):
        return self.get_seeker_state(target_state, np.array([0., 0., 0., 0., 0., 0.]))


if __name__ == '__main__':
    missile = Missile3D()
    for i in range(1200):
        res = missile.missile_step(ego_fighter_state=np.array([3000, 3000, 1500, 200, 0, 0]),
                                   enm_fighter_state=np.array([6000, 6000, 1000, 200, 0, 0]))
        m, info = res
        print(info)
        if info['hit'] or info['crash']:
            break
