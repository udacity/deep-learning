import numpy as np
import csv


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def earth_to_body_frame(ii, jj, kk):
    # C^b_n
    R = [[C(kk) * C(jj), C(kk) * S(jj) * S(ii) - S(kk) * C(ii), C(kk) * S(jj) * C(ii) + S(kk) * S(ii)],
         [S(kk) * C(jj), S(kk) * S(jj) * S(ii) + C(kk) * C(ii), S(kk) * S(jj) * C(ii) - C(kk) * S(ii)],
         [-S(jj), C(jj) * S(ii), C(jj) * C(ii)]]
    return np.array(R)


def body_to_earth_frame(ii, jj, kk):
    # C^n_b
    return np.transpose(earth_to_body_frame(ii, jj, kk))


class PhysicsSim():
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5.):
        self.init_pose = init_pose
        self.init_velocities = init_velocities
        self.init_angle_velocities = init_angle_velocities
        self.runtime = runtime

        self.gravity = -9.81  # m/s
        self.rho = 1.2
        self.mass = 0.958  # 300 g
        self.dt = 1 / 50.0  # Timestep
        self.C_d = 0.3
        self.l_to_rotor = 0.4
        self.propeller_size = 0.1
        width, length, height = .51, .51, .235
        self.dims = np.array([width, length, height])  # x, y, z dimensions of quadcopter
        self.areas = np.array([length * height, width * height, width * length])
        I_x = 1 / 12. * self.mass * (height**2 + width**2)
        I_y = 1 / 12. * self.mass * (height**2 + length**2)  # 0.0112 was a measured value
        I_z = 1 / 12. * self.mass * (width**2 + length**2)
        self.moments_of_inertia = np.array([I_x, I_y, I_z])  # moments of inertia

        env_bounds = 300.0  # 300 m / 300 m / 300 m
        self.lower_bounds = np.array([-env_bounds / 2, -env_bounds / 2, 0])
        self.upper_bounds = np.array([env_bounds / 2, env_bounds / 2, env_bounds])

        self.reset()

    def reset(self):
        self.time = 0.0
        self.pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]) if self.init_pose is None else np.copy(self.init_pose)
        self.v = np.array([0.0, 0.0, 0.0]) if self.init_velocities is None else np.copy(self.init_velocities)
        self.angular_v = np.array([0.0, 0.0, 0.0]) if self.init_angle_velocities is None else np.copy(self.init_angle_velocities)
        self.linear_accel = np.array([0.0, 0.0, 0.0])
        self.angular_accels = np.array([0.0, 0.0, 0.0])
        self.prop_wind_speed = np.array([0., 0., 0., 0.])
        self.done = False

    def find_body_velocity(self):
        body_velocity = np.matmul(earth_to_body_frame(*list(self.pose[3:])), self.v)
        return body_velocity

    def get_linear_drag(self):
        linear_drag = 0.5 * self.rho * self.find_body_velocity()**2 * self.areas * self.C_d
        return linear_drag

    def get_linear_forces(self, thrusts):
        # Gravity
        gravity_force = self.mass * self.gravity * np.array([0, 0, 1])
        # Thrust
        thrust_body_force = np.array([0, 0, sum(thrusts)])
        # Drag
        drag_body_force = -self.get_linear_drag()
        body_forces = thrust_body_force + drag_body_force

        linear_forces = np.matmul(body_to_earth_frame(*list(self.pose[3:])), body_forces)
        linear_forces += gravity_force
        return linear_forces

    def get_moments(self, thrusts):
        thrust_moment = np.array([(thrusts[3] - thrusts[2]) * self.l_to_rotor,
                            (thrusts[1] - thrusts[0]) * self.l_to_rotor,
                            0])# (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1]) * self.T_q])  # Moment from thrust

        drag_moment =  self.C_d * 0.5 * self.rho * self.angular_v * np.absolute(self.angular_v) * self.areas * self.dims * self.dims
        moments = thrust_moment - drag_moment # + motor_inertia_moment
        return moments

    def calc_prop_wind_speed(self):
        body_velocity = self.find_body_velocity()
        phi_dot, theta_dot = self.angular_v[0], self.angular_v[1]
        s_0 = np.array([0., 0., theta_dot * self.l_to_rotor])
        s_1 = -s_0
        s_2 = np.array([0., 0., phi_dot * self.l_to_rotor])
        s_3 = -s_2
        speeds = [s_0, s_1, s_2, s_3]
        for num in range(4):
            perpendicular_speed = speeds[num] + body_velocity
            self.prop_wind_speed[num] = perpendicular_speed[2]

    def get_propeler_thrust(self, rotor_speeds):
        '''calculates net thrust (thrust - drag) based on velocity
        of propeller and incoming power'''
        thrusts = []
        for prop_number in range(4):
            V = self.prop_wind_speed[prop_number]
            D = self.propeller_size
            n = rotor_speeds[prop_number]
            J = V / n * D
            # From http://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
            C_T = max(.12 - .07*max(0, J)-.1*max(0, J)**2, 0)
            thrusts.append(C_T * self.rho * n**2 * D**4)
        return thrusts

    def next_timestep(self, rotor_speeds):
        self.calc_prop_wind_speed()
        thrusts = self.get_propeler_thrust(rotor_speeds)
        self.linear_accel = self.get_linear_forces(thrusts) / self.mass

        position = self.pose[:3] + self.v * self.dt + 0.5 * self.linear_accel * self.dt**2
        self.v += self.linear_accel * self.dt

        moments = self.get_moments(thrusts)

        self.angular_accels = moments / self.moments_of_inertia
        angles = self.pose[3:] + self.angular_v * self.dt + 0.5 * self.angular_accels * self.angular_accels * self.dt ** 2
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        self.angular_v = self.angular_v + self.angular_accels * self.dt

        new_positions = []
        for ii in range(3):
            if position[ii] <= self.lower_bounds[ii]:
                new_positions.append(self.lower_bounds[ii])
                self.done = True
            elif position[ii] > self.upper_bounds[ii]:
                new_positions.append(self.upper_bounds[ii])
                self.done = True
            else:
                new_positions.append(position[ii])

        self.pose = np.array(new_positions + list(angles))
        self.time += self.dt
        if self.time > self.runtime:
            self.done = True
        return self.done
