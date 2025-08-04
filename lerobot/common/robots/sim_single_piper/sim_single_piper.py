#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from easydict import EasyDict
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
import numpy as np
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_sim_single_piper import SimSinglePiperConfig
# from env.single_arm_env import SingleArmEnv
# from env.sim_single_env import MujocoGSGymEnv
from env.sim_single_piper_env import SinglePiperEnv
import torch

# piper sdk
from piper_sdk import *

logger = logging.getLogger(__name__)


class SimSinglePiper(Robot):
    """
    Designed by cfy, jzh
    """

    config_class = SimSinglePiperConfig
    name = "single_piper"

    def __init__(self, config: SimSinglePiperConfig, mode: str):
        super().__init__(config)
        self.config = config
        self.mode = mode
        # 创建 motor 映射
        self.motors = {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
            "gripper.pos": 0.0,
        }
        # 创建 robot 是否连接的标志位
        self.is_robot_connected = False
        
        # create env
        self.cfg = EasyDict({
            "mujoco_config": {
                "model_path": "/home/ubuntu/mujoco_il_rl/model_assets/piper_on_desk/scene.xml",
            },
            "sim_mode": "inference",
            "camera_names": ["wrist_cam", "3rd"],
        })

        self.cameras = self.cfg["camera_names"]
        self.env = SinglePiperEnv(self.cfg)


    @property
    def _motors_ft(self) -> dict[str, type]:
        return {k: float for k in self.motors.keys()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (480, 640, 3) for cam in self.cfg["camera_names"]
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.is_robot_connected 

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        # 如果 robot 和 cam 都已成功连接, 则报错
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        
        if self.env is not None:
            self.is_robot_connected = True
        else:
            raise "env is not successful initialized!"
        # for cam in self.cameras.values():
        #     cam.connect()
        # logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        # 暂时空实现
        pass

    def configure(self):
        # 暂时空实现
        pass
    
    def sim_action(self, sim_action):
        pass
        # print(" in sim action")
        # if isinstance(sim_action, torch.Tensor):
        #     sim_action = sim_action.detach().cpu().numpy()
        # self.env.step(sim_action)

    def get_observation(self) -> dict[str, Any]:
        observation_images = self.env.get_observation()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        # print("observation_images",observation_images)
        # ==== 左臂 ====
        # 1) 拿出 joint_list
        robot_data = observation_images['robot'][0]
        joint_list = robot_data['joint_list']

        # 2) 只取前 6 个
        for i in range(6):
            pos = joint_list[i]['joint_pos']
            # round 保留 8 位小数（按你原来那个风格）
            self.motors[f"joint_{i+1}.pos"] = round(pos, 8)

        # 如果还要 gripper 的话，比如 joint_list[6] 是 link7，joint_list[7] 是 link8，
        # 假设 gripper 在 list 的最后一位，可以这样：
        gripper_pos = abs(joint_list[7]['joint_pos'])
        if gripper_pos > 1:
            gripper_pos = 1.0
        # print("gripper_pos",round(gripper_pos, 8))
        self.motors["gripper.pos"] = round(gripper_pos, 8)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        obs_dict = self.motors.copy()

        for cam_name in self.cfg["camera_names"]:
            start = time.perf_counter()
            try:
                img = observation_images["mujoco_img"][cam_name]
            except KeyError:
                raise ValueError(f"Camera {cam_name} not found in environment")
            obs_dict[cam_name] = img
            dt_ms = (time.perf_counter() - start) * 1e3
            # logger.info(f"{self} read {cam_name}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.motors is None:
            raise DeviceNotConnectedError(f"self motor value is None.")
        
        action_dict = self.motors.copy()
        return action_dict


    def disconnect(self):
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        logger.info(f"{self} disconnected.")
        self.env.close()
