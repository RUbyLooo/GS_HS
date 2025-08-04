import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import gym
from gym import spaces
import torch
import queue

import time
import threading
import abc
from typing import Any, Type, TypeVar
import copy

# GS 渲染器
from viewer.gs_render.gaussian_renderer import GSRenderer

# 可视化交互
import rerun as rr
import glfw

# 导入激光雷达包装类和扫描模式生成函数
from mujoco_lidar.scan_gen import (
    generate_HDL64,          # Velodyne HDL-64E 模式
    generate_vlp32,          # Velodyne VLP-32C 模式
    generate_os128,          # Ouster OS-128 模式
    LivoxGenerator,          # Livox系列雷达
    generate_grid_scan_pattern  # 自定义网格扫描模式
)
from mujoco_lidar.lidar_wrapper import MjLidarWrapper
from mujoco_lidar.scan_gen import generate_grid_scan_pattern



class BaseMujocoGSWorker():
    """
    Base class for mujoco-gs worker.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        rr.init("sim render", spawn=True)

        # 确保 worker running
        self.running = True
        # 仿真数据队列，最大容量1，保持最新状态
        self.sim_state_gs_queue = queue.Queue(maxsize=1)
        # 物理数据 + gs 渲染数据队列，最大容量1，保持最新状态
        self.phys_state_gs_queue = queue.Queue(maxsize=1)
        
        # mujoco 物理仿真线程
        self.model = mujoco.MjModel.from_xml_path(self.config.mujoco_config.model_path)
        self.data = mujoco.MjData(self.model)
        # gs 渲染线程
        self.gs_model_dict = self.config.gs_config.gs_model_dict
        self.gs_rgb_width = self.config.gs_config.gs_rgb_width
        self.gs_rgb_height = self.config.gs_config.gs_rgb_height
        self.gs_renderer = GSRenderer(self.gs_model_dict, self.gs_rgb_width, self.gs_rgb_height)
        self.gs_thread = threading.Thread(target=self._gs_render_loop)
        self.gs_thread.start()
        self.is_reset = True

        # 自由相机渲染 (optional)
        if self.config.enable_free_camera == True:
            self.lock = threading.Lock()
            self.mouse_dx = 0
            self.mouse_dy = 0
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self.free_camera = mujoco.MjvCamera()
            self.vopt = mujoco.MjvOption()
            self.pert = mujoco.MjvPerturb()

            self.free_cam_fovy = 60
            self.free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            mujoco.mjv_defaultFreeCamera(self.model, self.free_camera)
            self.free_camera.azimuth = -90.0      # 水平角度，可以自己调
            self.free_camera.elevation = 0.0    # 抬头看，正数 30 度
            self.free_camera.distance = 1.0
            self.free_camera.lookat = np.array([2.0, 1.0, 1.0], dtype=np.float64)

            # ✅ 关键，必须用 self.vopt 和 self.pert
            mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.pert,
                                self.free_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

            self.mouse_pressed = {'left': False, 'right': False}
            self.mouse_pos = {'x': 0, 'y': 0}

            # GLFW初始化和窗口创建必须在主线程
            if not glfw.init():
                raise RuntimeError("Failed to initialize glfw")
            self.window = glfw.create_window(400, 300, "Free Camera", None, None)
            if not self.window:
                glfw.terminate()
                raise RuntimeError("Failed to create glfw window")
            glfw.make_context_current(self.window)

            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
            glfw.set_scroll_callback(self.window, self._on_mouse_scroll)

        # mid 360 lidar (optional)
        if self.config.enable_mid360_lidar == True:
            self.mujoco_lock = threading.Lock()
            # self.global_map = self.load_point_cloud("/home/cfy/cfy/cfy/lerobot_nn/mobile_ai_rl/mobile_ai_gs/models/3dgs/scene/global_map.pcd")
            ### livox lidar 相关
            self.livox_generator = LivoxGenerator("mid360")  # 可选: "avia", "mid40", "mid70", "mid360", "tele"
            self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            self.lidar_sim = MjLidarWrapper(
                self.model, 
                self.data, 
                site_name="lidar_site",  # 与MJCF中的<site name="...">匹配
                args={
                    "geom_group": [0],  # 根据你想检测的 group 设置
                    "enable_profiling": False, # 启用性能分析（可选）
                    "verbose": False           # 显示详细信息（可选）
                }
            )

            self.localization_pose = {
                "position": np.zeros(3),
                "orientation": np.array([0, 0, 0, 1])  # quaternion wxyz
            }
        

    def _on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed['left'] = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.mouse_pressed['right'] = (action == glfw.PRESS)

    def _on_mouse_move(self, window, xpos, ypos):
        dx = xpos - self.mouse_pos['x']
        dy = ypos - self.mouse_pos['y']

        with self.lock:
            self.mouse_dx += dx
            self.mouse_dy += dy

        self.mouse_pos['x'] = xpos
        self.mouse_pos['y'] = ypos

    def _on_mouse_scroll(self, window, xoffset, yoffset):
        with self.lock:
            # 这里调正符号，滚轮向前缩小distance
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                0, yoffset / 10.0, self.scene, self.free_camera)

    def glfw_loop_step(self):
        glfw.poll_events()
        glfw.swap_buffers(self.window)

        with self.lock:
            dx = self.mouse_dx
            dy = self.mouse_dy
            self.mouse_dx = 0
            self.mouse_dy = 0

            if self.mouse_pressed['left']:
                mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                                    dy / 300, dx / 300, self.scene, self.free_camera)

            elif self.mouse_pressed['right']:
                self._custom_move_camera(dx, dy)
            mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.pert,
                                self.free_camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
    
    def _custom_move_camera(self, dx, dy):
        # 计算相机的右向和上向向量
        az = np.deg2rad(self.free_camera.azimuth)
        el = np.deg2rad(self.free_camera.elevation)

        forward = np.array([
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el)
        ])

        up_world = np.array([0, 0, 1])
        if np.allclose(np.abs(np.dot(forward, up_world)), 1.0):
            up_world = np.array([0, 1, 0])

        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # 计算平移向量，比例自己调节
        move_speed = 0.01
        move_vec = right * (dx * move_speed) + up * (-dy * move_speed)  # dy取反符合屏幕方向

        # 更新相机目标点
        self.free_camera.lookat += move_vec
    
    
    def _get_free_camera_pose(self):
        lookat = np.array(self.free_camera.lookat)
        distance = self.free_camera.distance
        azimuth = np.deg2rad(self.free_camera.azimuth)
        elevation = np.deg2rad(self.free_camera.elevation)

        forward = np.array([
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation)
        ])

        camera_pos = lookat - distance * forward

        up_world = np.array([0, 0, 1])
        if np.allclose(np.abs(np.dot(forward, up_world)), 1.0):
            up_world = np.array([0, 1, 0])

        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        rot_mat = np.column_stack((right, up, -forward))
        quat = R.from_matrix(rot_mat).as_quat()
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

        return camera_pos, quat_wxyz


    def _phy_simulation_step(self):
        if self.running:
            # mujoco 物理仿真向前一步
            mujoco.mj_step(self.model, self.data)
            # 取出当前观测, 放入队列中
            sim_data = self._get_simulation_data()

            # 尝试放入队列，放不进去就先取出旧数据，保证放入最新数据
            try:
                self.sim_state_gs_queue.put_nowait(sim_data)
            except queue.Full:
                try:
                    _ = self.sim_state_gs_queue.get_nowait()
                    self.sim_state_gs_queue.put_nowait(sim_data)
                except queue.Empty:
                    pass


            time.sleep(0.002)  # 模拟高频仿真循环

    def _get_cur_obs(self) -> dict[str, Any] | None:
        if self.running:
            if self.is_reset == True:
                while True:
                    try:
                        # 阻塞式获取数据，无限等待，直到取到数据为止
                        cur_obs = self.phys_state_gs_queue.get(timeout=1.0)
                        if cur_obs is not None:
                            return cur_obs
                    except queue.Empty:
                        # 超时继续等，不返回 None
                        continue

            else:
                try:
                    # 等待最新仿真数据，超时则返回空
                    cur_obs = self.phys_state_gs_queue.get(timeout=1.0)
                    return cur_obs
                except queue.Empty:
                    return None

    def _set_cur_obs(
        self,
        sim_data: dict[str, Any],
        gs_img  : dict[str, Any],
    ) -> None:
        phys_state_gs_data = {}

        # 提取 robot 数据
        phys_state_gs_data = {
            "robot": sim_data.get("robot", []),  # 提取所有 robot 信息
            "gs_img": gs_img  # 加入 gs 渲染结果
        }

        # 尝试放入队列，放不进去就先取出旧数据，保证放入最新数据
        try:
            self.phys_state_gs_queue.put_nowait(phys_state_gs_data)
        except queue.Full:
            try:
                _ = self.phys_state_gs_queue.get_nowait()
                self.phys_state_gs_queue.put_nowait(phys_state_gs_data)
            except queue.Empty:
                pass


    def _gs_render_loop(self):
        while self.running:
            try:
                # step 1 : 等待最新仿真数据，超时则继续等待
                sim_data = self.sim_state_gs_queue.get(timeout=0.01)
                # step 2 : 取出最新队列中的仿真数据, 进行 gs 渲染更新
                gs_img = self._render(sim_data)
                # step 3 : 将最新的物理仿真数据与 gs 渲染数据组合成当前观测
                self._set_cur_obs(sim_data, gs_img)
            except queue.Empty:
                # 超时没拿到数据，可以选择继续等待或者做空渲染
                continue

    def _lidar_render(self):
        # 更新场景并发布雷达点云
        rays_theta, rays_phi = self.livox_generator.sample_ray_angles()
        with self.mujoco_lock:
            self.lidar_sim.update_scene(self.model, self.data)
        points = self.lidar_sim.get_lidar_points(rays_phi, rays_theta, self.data)

        # 提取 XYZ
        xyz = points[:, :3]

        # 以 Z 值归一化为颜色（模拟 RViz 效果）
        z = xyz[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-6)  # 避免除0

        # 颜色映射为蓝->红（冷暖色）
        # 你可以用 matplotlib colormap 或简单手动插值
        colors = np.stack([
            z_norm * 255,  # R通道：低z高红
            (1.0 - np.abs(z_norm - 0.5) * 2) * 255,      # G通道为0
            (1.0 - z_norm) * 255          # B通道：高z高蓝
        ], axis=1).astype(np.uint8)

        # rerun 显示
        rr.log("lidar/points", rr.Points3D(positions=xyz, colors=colors))


    def _lidar_render_loop(self):
        if self.running:
            self._lidar_render()
            time.sleep(0.5)

    
    def _update_gs_scene(
        self,
        sim_data: dict[str, Any],
    ) -> None:
        """
        利用 mujoco 仿真数据，更新 gs 中的机器人关节与物体位姿，支持多机器人和多物体。
        """

        # 防止数据突变，拷贝一份
        sim_data_safe = copy.deepcopy(sim_data)

        # step 1 : 更新所有机器人关节 pose
        if "robot" in sim_data_safe:
            for robot in sim_data_safe["robot"]:
                joint_list = robot.get("joint_list", [])

                for joint in joint_list:
                    joint_name = joint["name"]
                    joint_pos = np.array(joint["pos"])
                    joint_quat = np.array(joint["quat_wxyz"])

                    # 更新 gs 中机器人每个关节的位姿
                    self.gs_renderer.set_obj_pose(joint_name, joint_pos, joint_quat)

        # step 2 : 更新所有物体
        if "item" in sim_data_safe:
            for item in sim_data_safe["item"]:
                item_name = item["name"]
                item_pos = np.array(item["pos"])
                item_quat = np.array(item["quat_wxyz"])

                # 更新 gs 中物体的位姿
                self.gs_renderer.set_obj_pose(item_name, item_pos, item_quat)

        def multiple_quaternion_vector3d(qwxyz, vxyz):
            qw = qwxyz[..., 0]
            qx = qwxyz[..., 1]
            qy = qwxyz[..., 2]
            qz = qwxyz[..., 3]
            vx = vxyz[..., 0]
            vy = vxyz[..., 1]
            vz = vxyz[..., 2]
            qvw = -vx*qx - vy*qy - vz*qz
            qvx =  vx*qw - vy*qz + vz*qy
            qvy =  vx*qz + vy*qw - vz*qx
            qvz = -vx*qy + vy*qx + vz*qw
            vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
            vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
            vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
            return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)
        
        def multiple_quaternions(qwxyz1, qwxyz2):
            q1w = qwxyz1[..., 0]
            q1x = qwxyz1[..., 1]
            q1y = qwxyz1[..., 2]
            q1z = qwxyz1[..., 3]

            q2w = qwxyz2[..., 0]
            q2x = qwxyz2[..., 1]
            q2y = qwxyz2[..., 2]
            q2z = qwxyz2[..., 3]

            qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
            qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
            qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
            qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

            return torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)
        
        self.gs_renderer.renderer.need_rerender = True
        self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
        self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    def _get_gs_img(
        self,
        sim_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        获取 gs 中图像数据，支持多相机输入
        """
        if "camera" not in sim_data:
            return None
        
        # 防止数据突变，拷贝一份
        sim_data_safe = copy.deepcopy(sim_data)

        gs_img_result = {}
        # step 1 : 更新自由相机 (optional)
        if self.config.enable_free_camera == True:
            free_cam_pos, free_cam_quat = self._get_free_camera_pose()
            free_cam_fovy = self.free_cam_fovy

            self.gs_renderer.set_camera_fovy(free_cam_fovy * np.pi / 180.)
            self.gs_renderer.set_camera_pose(free_cam_pos, free_cam_quat)
            rgb_img, depth_img = self.gs_renderer.render()

            gs_img_result["free_camera"] = {
                "rgb": rgb_img,
                "depth": depth_img
            }
        # step 2 : 更新 gs 环境中所有的相机位姿
        for cam_info in sim_data_safe["camera"]:
            cam_name = cam_info["name"]
            cam_fovy = cam_info["fovy"]
            cam_pos = np.array(cam_info["pos"])
            cam_quat = np.array(cam_info["quat"])

            # 设置 gs 相机参数并渲染
            self.gs_renderer.set_camera_fovy(cam_fovy * np.pi / 180.)
            self.gs_renderer.set_camera_pose(cam_pos, cam_quat)
            rgb_img, depth_img = self.gs_renderer.render()

            # 存储结果
            gs_img_result[cam_name] = {
                "rgb": rgb_img,
                "depth": depth_img
            }

        return gs_img_result
        

    def _render(
        self,
        sim_data: dict[str, Any],
        ) -> dict[str, Any] | None:
        # step 1 : 更新 gs 场景
        self._update_gs_scene(sim_data)
        # step 2 : 返回图像渲染结果
        gs_img_result = self._get_gs_img(sim_data)
        # 可视化图像与关节角
        if self.config.display_data:
            for cam_name, cam_data in gs_img_result.items():
                rgb_img = cam_data.get("rgb", None)
                if rgb_img is not None:
                    rr.log(f"{cam_name}.rgb", rr.Image(rgb_img), static=True)
                if self.config.enable_depth_camera:
                    depth_img = cam_data.get("depth", None)
                    if depth_img is not None:
                        rr.log(f"{cam_name}.depth", rr.Image(depth_img), static=True)
            # 记录机器人关节角度
            for robot in sim_data.get("robot", []):
                robot_name = robot["name"]
                for joint in robot.get("joint_list", []):
                    joint_name = joint["name"]
                    joint_pos = joint["joint_pos"]  # 只记录关节角度
                    rr.log(f"{robot_name}.{joint_name}.pos", rr.Scalar(joint_pos))

        return gs_img_result


    def _print_all_joint_info(self):
        """
        打印模型中所有关节的名称、ID、范围限制和当前qpos值
        """
        print("\n=== 关节信息 ===")
        print(f"{'Joint Name':<20} {'Type':<15} {'Qpos Addr':<10} {'Range':<25} {'Current Value':<15}")
        print("-" * 90)
        
        for joint_id in range(self.model.njnt):
            # 获取关节名称
            name_addr = self.model.name_jntadr[joint_id]
            joint_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
            
            # 获取关节类型
            joint_type = self.model.jnt_type[joint_id]
            type_names = {
                0: "自由关节(6DOF)",
                1: "球关节(3DOF)", 
                2: "滑动关节",
                3: "铰链关节"
            }
            type_str = type_names.get(joint_type, "未知类型")
            
            # 获取qpos地址和范围
            qpos_addr = self.model.jnt_qposadr[joint_id]
            if self.model.jnt_limited[joint_id]:
                jnt_range = f"[{self.model.jnt_range[joint_id,0]:.2f}, {self.model.jnt_range[joint_id,1]:.2f}]"
            else:
                jnt_range = "无限制"
            
            # 获取当前值
            if joint_type == 0:  # 自由关节
                current_val = self.data.qpos[qpos_addr:qpos_addr+7]
            elif joint_type == 1:  # 球关节
                current_val = self.data.qpos[qpos_addr:qpos_addr+4]
            else:  # 滑动/铰链关节
                current_val = self.data.qpos[qpos_addr]
            
            print(f"{joint_name:<20} {type_str:<15} {qpos_addr:<10} {jnt_range:<25} {str(current_val):<15}")


    def _print_all_body_info(self):
        """
        打印模型中所有body的名称、ID、位置和四元数姿态信息
        """
        print("\n=== Body 信息 ===")
        print(f"{'Body Name':<25} {'Body ID':<8} {'Position':<30} {'Quaternion':<35}")
        print("-" * 100)
        
        for body_id in range(self.model.nbody):
            # 获取body名称
            name_addr = self.model.name_bodyadr[body_id]
            body_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
            
            # 获取位置和四元数
            pos = self.data.body(body_id).xpos
            quat = self.data.body(body_id).xquat
            
            print(f"{body_name:<25} {body_id:<8} {str(pos):<30} {str(quat):<35}")


    def _get_joint_qpos(self, joint_name: str):
        """
        通过关节名称获取当前qpos值
        :param joint_name: 关节名称字符串
        :return: 对应的qpos值(标量或数组)
        """
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"未找到名为 '{joint_name}' 的关节")
        
        qpos_addr = self.model.jnt_qposadr[joint_id]
        joint_type = self.model.jnt_type[joint_id]
        
        if joint_type == 0:  # 自由关节
            return self.data.qpos[qpos_addr:qpos_addr + 7]  # 位置(3) + 四元数(4)
        elif joint_type == 1:  # 球关节
            return self.data.qpos[qpos_addr:qpos_addr + 4]  # 四元数(4)
        else:  # 滑动/铰链关节
            return self.data.qpos[qpos_addr]  # 标量值


    def _get_cur_arm_joints_gripper_state(self, arm_prefix: str = "left") -> tuple[np.ndarray, float]:
        """
        获取机械臂当前 6 个关节角度和夹爪开合状态

        参数:
            arm_prefix (str): 机械臂关节前缀（如 'left', 'right'）

        返回:
            arm_cur_joint_pos: np.ndarray，6个关节角度（单位：rad）
            gripper_state: float，夹爪开合程度（归一化 0~1）
        """
        # 构造关节名列表
        if arm_prefix == "single":
            joint_names = [f"joint{i}" for i in range(1, 7)]
            gripper_joint7 = f"joint7"
            gripper_joint8 = f"joint8"
        elif arm_prefix is None:
            raise "arm_prefix not set"
        else:
            joint_names = [f"{arm_prefix}_joint{i}" for i in range(1, 7)]
            gripper_joint7 = f"{arm_prefix}_joint7"
            gripper_joint8 = f"{arm_prefix}_joint8"

        # 读取6个主关节位置
        arm_cur_joint_pos = np.array([self.get_joint_qpos(name) for name in joint_names])

        # 获取两个 gripper 关节状态（可做校验是否对称）
        gripper_7 = self.get_joint_qpos(gripper_joint7)
        gripper_8 = self.get_joint_qpos(gripper_joint8)

        # 归一化夹爪开合状态（按你的约定 0.035 是最大开合量的一半）
        gripper_state = gripper_7 / 0.035

        return arm_cur_joint_pos, gripper_state
        

    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过 body 名称获取其位姿信息, 返回一个7维向量
            :param body_name: body名称字符串
            :return: 7维numpy数组, 格式为 [x, y, z, w, x, y, z]
            :raises ValueError: 如果找不到指定名称的body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        
        # 提取位置和四元数并合并为一个7维向量
        position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]
        
        return position, quaternion
    
    def _get_geom_pose(self, geom_name: str):
        """
        通过 geom 名称获取其位姿信息
        Args:
            geom_name : 几何体名字
        Return:
            pos       : 几何体位置 (世界系)
            quat_wxyz : 几何体姿态 (世界系)
        """
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f"未找到名为 '{geom_name}' 的geom")

        pos = self.data.geom_xpos[geom_id]
        rot_mat = np.array(self.data.geom_xmat[geom_id]).reshape(3, 3)
        quat_xyzw = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return pos, np.array(quat_wxyz)
    
    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        通过 site 名称获取其位姿信息
        Args:
            site_name : site 名字
        Return:
            pos       : site 位置 (世界系)
            quat_wxyz : site 姿态 (世界系)
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 {site_name} 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos)        # shape (3,)

        # 姿态
        xmat = np.array(self.data.site(site_id).xmat)            # shape (9,)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)                    # [w, x, y, z]

        return position, quaternion
    
    def _get_sensor_data(self, sensor_name: str):
        """
        通过 sensor 名称获取传感器数据
        Args:
            sensor_name     : sensor 名字
        Return:
            sensor_values   : sensor 值
        """
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]
        return sensor_values



    def stop(self):
        self.running = False
        # 关闭 gs 渲染
        if hasattr(self, "gs_renderer") and self.gs_renderer is not None:
            try:
                self.gs_renderer.finish()
            except Exception:
                pass
            self.gs_renderer = None
        self.gs_thread.join()


    def _get_simulation_data(self) -> dict:
        """
        Returns real-time sim data. 必须由子类实现.
        """
        raise NotImplementedError