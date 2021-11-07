import numpy as np
from collections import deque
from math import sin, cos, asin


class MissileConfig(object):
    def __init__(self):
        self.num_missile = 1              # []      飞机带弹数量
        self.K = 5
        self.dt = 1 / 12.                 # [s]
        self.missile_vel = 400            # [m/s]   导弹-初始速度（）
        self.max_missile_acc = 300        # []      导弹-机动的最大加速度（近距不考虑衰减）

        self.shoot_max_distance = 6000    # [m]     规则-最远发弹距离   6000
        self.shoot_max_angle = 60         # [deg]   规则-发弹的最大方位角  60
        self.shoot_lock_time = 0.5          # [s]     规则-持续锁定1s发弹

        self.hit_distance = 100            # [m]     导弹-命中敌机的判定条件
        self.missile_last_time = 25       # [s]     导弹-最大飞行时长
        self.flag_render = True


class MissileCore(object):
    """
    take as input the enemy's fighter and the missile's state.
    Notice that both of the information take the form (x, y, z, V, psi-heading, gma-climbing)
    """
    def __init__(self):
        self.args = MissileConfig()

    def missile_accel(self, missile_state, commanded_accel):
        """

        :param missile_state:
        :param commanded_accel:
        :return:
        """
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
                         ax, ay / (Vm * cos(gma_m)), -az / Vm])  # 世界坐标系到导弹机体坐标系

    def _vect_cross(self, a, b):
        ax, ay, az = a
        bx, by, bz = b
        return np.array([ay*bz-az*by, az*bx-ax*bz, ax*by - ay*bx])

    def _constraint_acc(self, raw_acc_vector):
        a_value = np.linalg.norm(raw_acc_vector)
        if a_value > self.args.max_missile_acc:
            raw_acc_vector = raw_acc_vector * self.args.max_missile_acc / a_value
        return raw_acc_vector

    def _pn_guidance(self, seeker_state):
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
        return self._constraint_acc(ach)

    def step(self, missile_cur_state, target_cur_state):
        """
        target_state:   [x, y, z, V, psi--heading, gma-climbing]
        missile_state:  [x, y, z, V, psi-heading, gma-climbing]
        seeker_state:   [d_x, d_y, d_z, d_vx, d_vy, d_vz]
        """
        seeker_cur_state = self.get_seeker_state(target_cur_state, missile_cur_state)
        commanded_accel = self._pn_guidance(seeker_cur_state)
        missile_next_state = self._rK4_step(missile_cur_state, self.missile_accel,
                                            self.args.dt, params=commanded_accel)
        return missile_next_state

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

    def transfer2angles(self, pos_vel):
        vx, vy, vz = pos_vel[3:6]
        R = np.linalg.norm([vx, vy, vz])

        psi_rec = np.arctan2(vy, vx)  # yaw    ---- heading angle
        gma_rec = -asin(vz / R)       # pitch  ---- climb angle
        return [psi_rec, gma_rec]

    def transfer2raw(self, target_state):
        return self.get_seeker_state(target_state, np.array([0., 0., 0., 0., 0., 0.]))

    def determine_missile_crash(self, missile_ith):
        cur_time_step = missile_ith['step'] * self.args.dt
        flag_missile_crash = False
        if cur_time_step >= self.args.missile_last_time or np.sum(missile_ith['increment_distance']) >= 100:
            flag_missile_crash = True
        return flag_missile_crash

    def determine_hit(self, ego_missile_state, enm_fighter_state, missile_ith):
        distance = np.linalg.norm(ego_missile_state[:3] - enm_fighter_state[:3])
        missile_ith['increment_distance'].append(np.sign(distance - missile_ith['pre_distance']))
        missile_ith['pre_distance'] = distance
        flag_hit = False
        # print(f"{distance}")
        if distance <= self.args.hit_distance:
            flag_hit = True
        return flag_hit


class MissileRule(object):
    def __init__(self):
        self.args = MissileConfig()
        self.rule_info = {
            'lock_duration': deque(maxlen=int(self.args.shoot_lock_time / self.args.dt)),
        }

    def judge_shoot_condition(self, cur_time_step, ego_state, enm_state, missile_info):
        # 0) 不允许发弹
        if not missile_info[0]['allow_shoot']:
            return False
        # 1) 无弹可打
        num_all_missile = len(missile_info)
        num_available_missile = np.sum([1. - missile_info[_]['launched'] for _ in range(num_all_missile)])
        # print('available missile', num_available_missile)
        if num_available_missile <= 0:
            return False

        # 2) 发弹条件判断，（）
        distance = max(np.linalg.norm(ego_state[:3] - enm_state[:3]), 50) # max is to avoid 0 in denominator
        pos_vector = enm_state[:3] - ego_state[:3]
        vel_vector = ego_state[3:]
        ego_angle = np.rad2deg(np.arccos(np.sum(pos_vector * vel_vector) / (distance * np.linalg.norm(vel_vector))))
        shoot = 1 if ego_angle < self.args.shoot_max_angle else 0
        self.rule_info['lock_duration'].append(shoot)
        # 2.1) 发弹距离与锁定时长
        flag_shoot = np.sum(self.rule_info['lock_duration']) >= self.rule_info['lock_duration'].maxlen \
                     and distance <= self.args.shoot_max_distance

        # 3) 如果满足发弹条件，一枚一枚发
        if flag_shoot:
            for k in range(num_all_missile):
                # 3.1) 发弹间隔
                flag_shoot_interval_time = 0
                if k == 0:
                    flag_shoot_interval_time = 1
                else:
                    flag_shoot_interval_time = cur_time_step - missile_info[k-1]['launch_time'] >= 3 * 12

                if not missile_info[k]['launched'] and flag_shoot_interval_time:
                    missile_info[k]['launched'] = True
                    break


class Missile3D(object):
    def __init__(self, allow_shoot=True):
        super(Missile3D, self).__init__()
        self.simulator = MissileCore()
        self.args = self.simulator.args
        self.num_missile = self.args.num_missile
        self.missile_rule = MissileRule()
        self.missile_info = [{
            'launched': False, 'strike_enm_fighter': False, 'destructed': False, 'flying': False,
            'initial_state': None, 'current_state': None, 'launch_time': 0, 'step': 0,
            'pre_distance': self.args.shoot_max_distance,
            'trajectory': [],
            'increment_distance': deque(maxlen=13),
            'allow_shoot': allow_shoot
        } for _ in range(self.num_missile)]

    def _missile_initial_vel(self, ego_fighter_vel, max_missile_vel):
        return max_missile_vel + ego_fighter_vel

    def _initialize_missile_state(self, initial_missile, initial_target):
        """
        initial_missile:    [x_ego, y_ego, z_ego, vx_ego, vy_ego, vz_ego, Vmissile]
        initial_target:     [x_enm, y_enm, z_enm, vx_enm, vy_enm, vz_enm]
        """
        # 1) the target's states
        psi, gma = self.simulator.transfer2angles(initial_target)
        target_state = np.array([*initial_target[:3], np.linalg.norm(initial_target[3:6]), psi, gma])

        # 2) the missile's states
        psi_m_0, gma_m_0 = self.simulator.transfer2angles(initial_missile)
        missile_state = np.array([*initial_missile[:3], initial_missile[6], psi_m_0, gma_m_0])
        return missile_state, target_state

    def make_step(self, ego_fighter_state, enm_fighter_state, fighter_time_step):
        """
        ego_fighter_state:      [x, y, z, vx, vy, vz]    (unit: m, m/s)
        enm_fighter_state:      [x, y, z, vx, vy, vz]
        """
        # todo: shoot rule
        self.missile_rule.judge_shoot_condition(fighter_time_step, ego_fighter_state, enm_fighter_state,
                                                self.missile_info)
        # simulation
        flag_strike_enm = False
        for i in range(self.num_missile):
            # 1）如果第i枚弹没有发射或已经坠毁，不仿真；
            if self.missile_info[i]['launched'] is False or self.missile_info[i]['destructed']:
                continue
            self.missile_info[i]['step'] += 1
            if self.missile_info[i]['initial_state'] is None:
                # 2）若导弹刚发射，设置导弹的初始位姿与初始速度
                missile_vel = self._missile_initial_vel(np.linalg.norm(ego_fighter_state[3:]), self.args.missile_vel)
                initial_missile = np.array([*ego_fighter_state, missile_vel])
                missile_state, target_state = self._initialize_missile_state(initial_missile, enm_fighter_state)
                print('initial_missile_state', missile_state)
                self.missile_info[i]['initial_state'] = missile_state
                self.missile_info[i]['current_state'] = missile_state
                self.missile_info[i]['launch_time'] = fighter_time_step
                self.missile_info[i]['flying'] = True
                if self.args.flag_render:
                    self.missile_info[i]['trajectory'].append(missile_state)
            else:
                # 3）若导弹已在飞行过程中，获取敌机信息即可
                psi, gma = self.simulator.transfer2angles(enm_fighter_state)
                target_state = np.array([*enm_fighter_state[:3], np.linalg.norm(enm_fighter_state[3:]), psi, gma])

                missile_state = self.simulator.step(self.missile_info[i]['current_state'], target_state)
                if self.args.flag_render:
                    self.missile_info[i]['trajectory'].append(missile_state)
            # 4) update the information of the missile
            self.missile_info[i]['current_state'] = missile_state
            flag_hit = self.simulator.determine_hit(missile_state, enm_fighter_state, self.missile_info[i])
            self.missile_info[i]['strike_enm_fighter'] = flag_hit
            flag_crash = self.simulator.determine_missile_crash(self.missile_info[i])
            self.missile_info[i]['destructed'] = flag_crash or flag_hit
            self.missile_info[i]['flying'] = False if flag_hit or flag_crash else True
            if self.missile_info[i]['strike_enm_fighter']:
                flag_strike_enm = True
        return flag_strike_enm

    def check_no_missile_alive(self):
        for i in range(self.num_missile):
            if self.missile_info[i]['flying']:
                return False
        return True


if __name__ == '__main__':
    missile = Missile3D()
    for i in range(1200):
        s = missile.make_step(ego_fighter_state=np.array([3000, 3000, 1500, 200, 0, 0]),
                              enm_fighter_state=np.array([6000, 6000, 1000, 200, 0, 0]), fighter_time_step=i)
        for j in range(missile.args.num_missile):
            print(missile.missile_info[j])
            print('----------' * 7)
        print('==========' * 7)
        if s:
            break
