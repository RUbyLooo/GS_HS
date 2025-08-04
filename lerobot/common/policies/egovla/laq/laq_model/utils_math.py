import math
PI = math.pi
import numpy as np
from math import acos, atan2, sqrt, acos, pi, sin, cos

def rotation_matrix_to_euler(R):
    """从旋转矩阵计算欧拉角(ZYZ顺序)"""
    sin_theta = sqrt(R[2, 0] ** 2 + R[2, 1] ** 2)
    singular = sin_theta < 1e-6

    if not singular:
        theta = atan2(sin_theta, R[2, 2])
        phi = atan2(R[1, 2]/sin(theta), R[0, 2]/sin(theta))
        psi = atan2(R[2, 1]/sin(theta), -R[2, 0]/sin(theta))

    else:
        theta = 0
        phi = 0
        psi = atan2(-R[0, 1], R[0, 0])

    return np.array([phi, theta, psi])


def rotation_matrix_to_quaternion(R):
    """将3x3旋转矩阵转换为四元数(w, x, y, z顺序)"""
    q = np.zeros(4)
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S

    return q / np.linalg.norm(q)  # 归一化