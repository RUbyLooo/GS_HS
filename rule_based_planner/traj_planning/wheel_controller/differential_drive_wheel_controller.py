import numpy as np

from .wheel_controller import WheelController


class DifferentialDriveWheelController(WheelController):
    def __init__(self, r, w):
        super().__init__()
        self._r = r
        self._w = w
        self._vx_scale = 1.0
        self._wz_scale = 1.0

    def ctrl(self, vx, vy, wz):
        vx = vx * self._vx_scale
        wz = wz * self._wz_scale
        v_left = (vx - self._w * wz) / self._r
        v_right = (vx + self._w * wz) / self._r
        return np.array([v_left, v_right])
