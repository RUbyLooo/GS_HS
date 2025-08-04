import numpy as np
import math
PI = math.pi
from math import atan2, sqrt, acos, pi, sin, cos, atan
import numpy as np
from lerobot.common.policies.egovla.laq.laq_model.utils_math import rotation_matrix_to_euler, rotation_matrix_to_quaternion

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

class PiperArm:
    def __init__(self):
        # DH参数定义（单位：米/弧度）
        self.alpha = [0, -pi / 2, 0, pi / 2, -pi / 2, pi / 2]  # 扭转角
        self.a = [0, 0, 0.28503, -0.02198, 0, 0]  # 连杆长度
        self.d = [0.123, 0, 0, 0.25075, 0, 0.091]  # 连杆偏移
        self.theta_offset = [0, -172.2135102 * pi / 180, -102.7827493 * pi / 180, 0, 0, 0]  # 初始角度偏移
        # self.theta_offset = [0, -172.241 * pi / 180, -100.78 * pi / 180, 0, 0, 0]  # 初始角度偏移
        # self.l = 0.145 + 0.091 # 夹爪末端点 到 joint4
        self.l = 0.091  # 夹爪末端点 到 joint4
    def dh_transform(self, alpha, a, d, theta):
        """
        计算Denavit-Hartenberg标准参数的4x4齐次变换矩阵

        参数:
        alpha (float): 连杆扭转角 (绕x_(i-1)轴的旋转角，弧度)
        a (float)    : 连杆长度 (沿x_(i-1)轴的平移量, 米)
        d (float)    : 连杆偏移量 (沿z_i轴的平移量, 米)
        theta (float): 关节角 (绕z_i轴的旋转角, 弧度)

        返回:
        np.ndarray: 4x4齐次变换矩阵
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # 构建标准DH变换矩阵
        transform = np.array([
            [ct, -st, 0, a],
            [ca * st, ca * ct, -sa, -sa * d],
            [sa * st, sa * ct, ca, ca * d],
            [0, 0, 0, 1]
        ])
        return transform

    def dh_transform_test(self, alpha, a, d, theta):
        """
        计算Denavit-Hartenberg标准参数的4x4齐次变换矩阵

        参数:
        alpha (float): 连杆扭转角（绕x_(i-1)轴的旋转角，弧度）
        a (float): 连杆长度（沿x_(i-1)轴的平移量，米）
        d (float): 连杆偏移量（沿z_i轴的平移量，米）
        theta (float): 关节角（绕z_i轴的旋转角，弧度）

        返回:
        np.ndarray: 4x4齐次变换矩阵
        """
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        ct = np.cos(theta)
        st = np.sin(theta)


        R_alpha = np.array([
            [1,  0,   0, 0],
            [0, ca, -sa, 0],
            [0, sa,  ca, 0],
            [0,  0,   0, 1]
        ])

        T_a = np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        R_theta = np.array([
            [ct, -st, 0, 0],
            [st, ct, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T_d = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])

        # 构建标准DH变换矩阵
        transform_test = R_alpha @ T_a @ R_theta @ T_d

        return transform_test


    def forward_kinematics(self, joints):
        T_total = np.eye(4)
        for i in range(6):
            T = self.dh_transform(self.alpha[i], self.a[i], self.d[i], self.theta_offset[i] + joints[i])
            T_total = T_total @ T

        return T_total

    def forward_kinematics_sub(self, joints, end):
        # 0_T_end
        T_total = np.eye(4)
        for i in range(end):
            print("i alpha a d theta", self.alpha[i], self.a[i], self.d[i], self.theta_offset[i])
            T = self.dh_transform(self.alpha[i], self.a[i], self.d[i], self.theta_offset[i] + joints[i])
            T_total = T_total @ T

        return T_total

    def inverse_kinematics(self, T_target):
        """Pieper解法逆运动学求解"""
        # 计算 joint4 位置
        joint4_p = T_target @ np.array([0, 0, -self.l, 1], dtype=float)
        px, py, pz = joint4_p[0], joint4_p[1], joint4_p[2]

        # px = T_target[0, 3]
        # py = T_target[1, 3]
        # pz = T_target[2, 3]

        # 计算 link1 2 3 角度
        theta1 = atan2(py , px)
        T01 = self.dh_transform(self.alpha[0], self.a[0], self.d[0], theta1)

        # Convert P05 to frame 1
        P15 = np.linalg.inv(T01) @ np.array([px, py, pz, 1])
        x1, z1 = P15[0], P15[2]
        a1, a2 = self.a[2], self.a[3]
        d1, d2 = self.d[2], self.d[3]
        l1 = sqrt(a1 ** 2 + d1 ** 2)
        l2 = sqrt(a2 ** 2 + d2 ** 2)
        l3 = sqrt(x1 ** 2 + z1 ** 2)

        cos_phi3 = (l1**2 + l2**2 - l3**2) / (2.0 * l1 * l2)
        if abs(cos_phi3) > 1:
            print("no ik solution, fail theta 3")
            return
        phi3 = acos(cos_phi3)
        print("phi3 is ", phi3 / PI * 180)

        phi3 = atan2(sqrt(1 - cos_phi3 ** 2), cos_phi3)
        # print("phi3 is ", phi3 / PI * 180)
        gamma = atan2(abs(self.d[3]), abs(self.a[3]))
        # print("gamma is ", gamma / PI * 180)
        theta3 = -(gamma + phi3) - self.theta_offset[2]


        cos_phi2 = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
        if abs(cos_phi2) > 1:
            print("no ik solution, fail theta 2")
        phi2 = acos(cos_phi2)

        beta = atan(x1 / z1)
        if z1 > 0:
            theta2 = - (PI / 2 + phi2 - beta) - self.theta_offset[1]
        else:
            beta = atan(x1 / abs(z1))
            print("phi2", phi2 / 3.14 * 180)
            print("beta", beta / 3.14 * 180)
            theta2 = - (phi2 - (PI / 2 - beta)) - self.theta_offset[1]

        q_sol = [theta1, theta2, theta3, 0, 0, 0]

        # 计算link4 5 6 角度
        T03 = self.forward_kinematics_sub(q_sol , 3)
        R03 = T03[0:3, 0:3]
        R34d = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]])
        T34d = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        R03d = R03 @ R34d
        R06 = T_target[0:3, 0:3]
        R36 = R03d.T @ R06

        print("needed R36", R36)
        rx, ry, rz = rotation_matrix_to_euler(R36)
        q_sol = [theta1, theta2, theta3, rx, ry, rz]
        T34 = self.get_joint_tf(3, rx)
        T45 = self.get_joint_tf(4, ry)
        T56 = self.get_joint_tf(5, rz)
        T36 = T34d.T @ (T34 @ T45) @ T56
        print("REAL T36", T36)

        return q_sol

    def get_joint_tf(self, joint_idx, angle):
        """获取指定关节的变换矩阵"""
        transform = self.dh_transform(self.alpha[joint_idx], self.a[joint_idx], self.d[joint_idx], self.theta_offset[joint_idx] + angle)
        return transform

