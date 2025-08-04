import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import time
import threading
from typing import Any, Type, TypeVar
import copy
import queue
# from viewer.gs_render.gaussian_renderer import GSRenderer
import glfw
import math
from math import atan2, sqrt, acos, pi, sin, cos, atan
PI = math.pi
import mujoco.viewer


class BaseMujocoGSWorker():
    """
    Base class for mujoco-gs worker.
    """
    def __init__(self, config):
        # 传入配置文件
        self.config = config

        # 初始化 mujoco 环境
        self.model = mujoco.MjModel.from_xml_path(self.config.mujoco_config.model_path)
        self.data = mujoco.MjData(self.model)

        self.count = 0

        # (必选) 初始化 mujoco 物理线程
        self.desired_path = None  # 用于记录期望的动作路径
        self.start_physical_thread()

        self.item_name = "apple"

        # 开一个物理引擎的线程
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 3
        self.handle.cam.azimuth = 0
        self.handle.cam.elevation = -30
        self.opt = mujoco.MjvOption()

        # # (Optional) 初始化 gs 线程
        # if self.config.gs_config.enable:
        #     self.gs_renderer = GSRenderer(self.model, self.data, self.config.gs_config)
        #     self.gs_thread = threading.Thread(target=self.gs_renderer.run)
        #     self.gs_thread.start()

        # 物理线程接受 action 队列
        self.desired_action_queue = queue.Queue()
        self.desired_action_lock = threading.Lock()
        self.current_episode_done = False

        # 物理线程往外传输队列
        self.sim_data_queue = queue.Queue(maxsize=5)
        self.sim_data_lock = threading.Lock()

        ## 以下都是渲染图像相关
        # ✅ 正确使用 glfw（模块调用，不加 self.）
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(640, 480, "offscreen", None, None)
        glfw.make_context_current(self.window)

        # mujoco 相机数据相关
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.target_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_quat_wxyz = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.target_quat_xyzw = np.roll(self.target_quat_wxyz, -1)

        self.desired_action_idx = 0

    def sync(self):
        self.handle.sync()


    def _physical_step(self):
        """
        物理线程的主循环
        """

        self.desired_action_idx = 0
        while not self.stop_flag:  # 主循环可以被外部打断
            if self.physical_thread_start_running:
                if self.config.sim_mode == "record":
                    if self.desired_path is None:
                        raise ValueError("Desired path must be set in record mode.")

                    with self.physical_lock:
                        # 设置当前 desired action
                        action = self.desired_path[self.desired_action_idx]
                        if action is not None:
                            self.data.ctrl[:] = action

                        # 执行一次物理模拟
                        mujoco.mj_step(self.model, self.data)

                        is_reached_desired_action = self._is_reached_desired_action(action)
                        if is_reached_desired_action:
                            self.desired_action_idx += 1
                            if self.desired_action_idx >= len(self.desired_path):
                            # if self.desired_action_idx >= 100:

                                print("Reached the end of the desired path.")
                                self.current_episode_done = True
                                self.physical_thread_start_running = False  # ✅ 正确地暂停物理模拟
                                continue  # 不退出线程，只暂停

                        # 获取 sim_data
                        sim_data = self._get_simulation_data()
                        # print("sim_data",sim_data)
                        with self.sim_data_lock:  # 注意锁保护
                            if self.sim_data_queue.full():
                                self.sim_data_queue.get()  # 移除最旧的数据
                            self.sim_data_queue.put(copy.deepcopy(sim_data))

                elif self.config.sim_mode == "inference":
                    with self.physical_lock:
                        with self.desired_action_lock:
                            if not self.desired_action_queue.empty():
                                desired_action = self.desired_action_queue.queue[-1].copy()
                                if desired_action is not None:
                                    self.data.ctrl[:] = desired_action
                                    # print("desired_action",desired_action)
                                    self.data.ctrl[6] = desired_action[6] * 0.035
                        # 执行一次物理模拟
                        mujoco.mj_step(self.model, self.data)
                        self.sync()

                        # 获取 sim_data
                        sim_data = self._get_simulation_data()
                        # print(" sim_data for inference: ", sim_data)
                        with self.sim_data_lock:  # 注意锁保护
                            if self.sim_data_queue.full():
                                self.sim_data_queue.get()  # 移除最旧的数据
                            self.sim_data_queue.put(copy.deepcopy(sim_data))

                time.sleep(0.002)  # 控制节奏
            else:
                time.sleep(0.01)  # 空闲状态休眠，避免 CPU 空转
            
            
    def start_physical_thread(self):
        if not hasattr(self, 'pysical_thread') or not self.pysical_thread.is_alive():
            self.stop_flag = False
            self.physical_thread_start_running = False
            self.pysical_thread = threading.Thread(target=self._physical_step)
            self.pysical_thread.start()
            # print("Pysical thread is start")
            self.physical_lock = threading.Lock()

            self.desired_path = None  # 初始化 desired_path

    def set_desired_action(self, desired_action):
        """
        设置期望的动作
        """
        with self.desired_action_lock:
            if self.desired_action_queue.full():
                self.desired_action_queue.get()  # 移除最旧的数据
            self.desired_action_queue.put(copy.deepcopy(desired_action))
    def pause_physical_thread(self):
        self.physical_thread_start_running = False

    def resume_physical_thread(self):
        self.physical_thread_start_running = True

    def stop_physical_thread(self):
        self.stop_flag = True
        self.physical_thread_start_running = False
        if self.pysical_thread.is_alive():
            self.pysical_thread.join()

    def get_latest_sim_data(self):
        """
        安全获取 sim_data_queue 中的最新仿真数据。
        如果队列为空或线程未在运行状态，则返回 None。
        """
        if not self.physical_thread_start_running:
            return None

        with self.sim_data_lock:
            if self.sim_data_queue.empty():
                print(f" sim_data_queue.empty !!! ")
                return None

            # 取出所有数据，仅保留最新的一个
            latest_data = None
            if not self.sim_data_queue.empty():
                latest_data = self.sim_data_queue.get()

            return latest_data

    def _is_reached_desired_action(self, desired_action):
        """
        判断是否到达 desired action
        """
        # 这里可以根据具体的需求来判断是否到达 desired action
        # 例如，可以判断当前的控制信号与 desired action 的差值是否小于某个阈值
        return np.linalg.norm(self.data.qpos[:6] - desired_action[:6]) < 0.05

    def _get_simulation_data(self) -> dict:
        """
        Returns real-time sim data. 必须由子类实现.
        """
        raise NotImplementedError
    

    
    



    
    

