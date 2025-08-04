import sys
import os
os.environ["MUJOCO_GL"] = "egl"

# 添加上一级目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from viewer.mujoco_render import mujoco_viewer
import mujoco,time,threading
import numpy as np
import pinocchio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import transformations as tf
from scipy.spatial.transform import Rotation
import torch
import cv2
import rerun as rr

### 底盘控制器
from rule_based_planner.traj_planning.wheel_controller import DifferentialDriveWheelController

### 机械臂规划相关
import ikpy.chain
from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.piper import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)
from viewer.gs_render import MOBILE_AI_ASSERT_DIR
### gs 渲染相关
from viewer.gs_render.gaussian_renderer import GSRenderer

# 相机参数相关
from config.camera_config import CAMERA_CONFIG


class BaseEnv(mujoco_viewer.BaseViewer):
    def __init__(self, path, cfg):
        """
        path :       XML 模型路径(MJCF)
        distance :   相机距离
        azimuth :    水平旋转角度
        elevation :  俯视角度
        """
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path

        if cfg["is_have_moving_base"] == True:
            # 移动底盘控制器
            self.wheel_radius = cfg["moving_base"]["wheel_radius"]
            self.half_wheelbase = cfg["moving_base"]["half_wheelbase"]
            self.wheel_controller = DifferentialDriveWheelController(self.wheel_radius, self.half_wheelbase)

        if cfg["is_have_arm"] == True:
            # 创建机械臂逆运动学解算模型
            self.my_chain = ikpy.chain.Chain.from_urdf_file("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/ik_asserts/piper_n.urdf")
            # 创建机械臂规划模型
            self.model_roboplan, self.collision_model, visual_model = load_models(use_sphere_collisions=True)
            if self.collision_model is None:
                raise ValueError("collision_model is None — collision model must be loaded before proceeding.")
            
            add_self_collisions(self.model_roboplan, self.collision_model)
            add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.1)
            self.target_frame = "link6"
            np.set_printoptions(precision=3)
            self.distance_padding = 0.001

            self.robot_link_list = cfg["robot_link_list"]
            self.obj_list = cfg["obj_list"]
            # self.obj_list = ["cf2", "mobile_ai"]
            # self.geom_robot_list = []
            self.index = 0


        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        rr.init("inference", spawn=True)

        if cfg["is_use_gs_render"] == True:
            ### 创建 gs 渲染
            self.rgb_fovy = CAMERA_CONFIG["rgb"]["fovy"]
            self.rgb_fovx = CAMERA_CONFIG["rgb"]["fovx"]
            self.rgb_width = CAMERA_CONFIG["rgb"]["width"]
            self.rgb_height = CAMERA_CONFIG["rgb"]["height"]
            self.gs_model_dict = {}
            self.gs_model_dict["background"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/scene/1lou_0527_res.ply"
            self.gs_model_dict["left_link1"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link1_rot.ply"
            self.gs_model_dict["left_link2"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link2_rot.ply"
            self.gs_model_dict["left_link3"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link3_rot.ply"
            self.gs_model_dict["left_link4"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link4_rot.ply"
            self.gs_model_dict["left_link5"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link5_rot.ply"
            self.gs_model_dict["left_link6"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link6_rot.ply"
            self.gs_model_dict["left_link7"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link7_rot.ply"
            self.gs_model_dict["left_link8"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link8_rot.ply"

            self.gs_model_dict["right_link1"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link1_rot.ply"
            self.gs_model_dict["right_link2"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link2_rot.ply"
            self.gs_model_dict["right_link3"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link3_rot.ply"
            self.gs_model_dict["right_link4"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link4_rot.ply"
            self.gs_model_dict["right_link5"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link5_rot.ply"
            self.gs_model_dict["right_link6"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link6_rot.ply"
            self.gs_model_dict["right_link7"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link7_rot.ply"
            self.gs_model_dict["right_link8"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link8_rot.ply"

            self.gs_model_dict["cf2"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/quadrotor/quadrotor_res.ply"
            self.gs_model_dict["mobile_ai"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/chassis/1.ply"
            self.gs_model_dict["desk"] = "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/desk/1louzhuozi.ply"
            
            # self.gs_model_dict["mobilebase0_base"] = "robot/chassis/chassis_rot.ply"
            self.gs_renderer = GSRenderer(self.gs_model_dict, self.rgb_width, self.rgb_height)
            self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)

        self.episode_len = cfg["episode_len"]
        self.is_save_record_data = cfg["is_save_record_data"]
        self.camera_names = cfg["camera_names"]
        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in self.camera_names},
                'qpos': [],
                'actions': []
            }
        }
        self._camera_name2id = ["0", "1", "2"]
        self.step_number = 0
        self.goal_reached_count = 0

        # 打印当前场景 joint 和 body 信息
        self.print_all_joint_info()
        self.print_all_body_info()


    def add_gaussian_model(self, name: str, abs_path: str):
        """
        使用绝对路径添加新的 Gaussian 模型，同时注册到 gs_model_dict 和 renderer。
        :param name: 模型名称（例如 'apple'）
        :param abs_path: 绝对路径，例如 '/home/.../apple_res.ply'
        """
        assert os.path.isabs(abs_path), f"[add_gaussian_model_abs_path] Path must be absolute, got: {abs_path}"
        assert os.path.exists(abs_path), f"[add_gaussian_model_abs_path] File does not exist: {abs_path}"

        # 将绝对路径转为相对路径（如果可以）供 gs_model_dict 使用；否则保留绝对路径
        base_dir = os.path.join(MOBILE_AI_ASSERT_DIR, "3dgs")
        try:
            rel_path = os.path.relpath(abs_path, base_dir)
            if rel_path.startswith(".."):  # 不在 base_dir 内，保留绝对路径
                rel_path = abs_path
        except:
            rel_path = abs_path

        self.gs_model_dict[name] = rel_path
        self.gs_renderer.load_single_model(name, abs_path)
        print(f"[GSModel] Model '{name}' added successfully.")




    def close(self):
        if hasattr(self, "gs_renderer") and self.gs_renderer is not None:
            try:
                self.gs_renderer.finish()
            except Exception:
                pass
            self.gs_renderer = None

        # super().close()



    def print_all_joint_info(self):
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


    def print_all_body_info(self):
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

    
    def get_joint_qpos(self, joint_name: str):
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
        


    def get_cur_arm_joints_gripper_state(self, arm_prefix: str = "left") -> tuple[np.ndarray, float]:
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
    

        
    def get_body_qpos_qvel(self, body_name):
        """
        获取与某个 body 相关的 qpos 和 qvel 索引及其值。

        参数:
            model: mujoco.MjModel 对象
            data: mujoco.MjData 对象
            body_name: 要查询的 body 名称，例如 "cf2"

        返回:
            qpos_vals: 对应 qpos 中的值
            qvel_vals: 对应 qvel 中的值
            qpos_range: (start, end)
            qvel_range: (start, end)
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # 获取第一个关联的 joint id
        joint_id = self.model.body_jntadr[body_id]

        # 对应的 qpos/qvel 起始索引
        qpos_addr = self.model.jnt_qposadr[joint_id]
        qvel_addr = self.model.jnt_dofadr[joint_id]

        # 计算该 joint 对应的自由度数量
        if (joint_id + 1) < self.model.njnt:
            nq = self.model.jnt_qposadr[joint_id + 1] - qpos_addr
            nv = self.model.jnt_dofadr[joint_id + 1] - qvel_addr
        else:
            nq = self.model.nq - qpos_addr
            nv = self.model.nv - qvel_addr

        qpos_vals = self.data.qpos[qpos_addr: qpos_addr + nq].copy()
        qvel_vals = self.data.qvel[qvel_addr: qvel_addr + nv].copy()

        return qpos_vals, qvel_vals, (qpos_addr, qpos_addr + nq), (qvel_addr, qvel_addr + nv)
        

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过body名称获取其位姿信息, 返回一个7维向量
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
    
    def get_geom_pose(self, geom_name: str):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f"未找到名为 '{geom_name}' 的geom")

        pos = self.data.geom_xpos[geom_id]
        rot_mat = np.array(self.data.geom_xmat[geom_id]).reshape(3, 3)
        quat_xyzw = Rotation.from_matrix(rot_mat).as_quat()  # [x, y, z, w]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return pos, np.array(quat_wxyz)
    
    def get_body_velocity(self, body_name: str):
        """
        根据 body 名称获取其世界系线速度和角速度
        -------------------------------------------------
        参数:
            body_name (str): MuJoCo 中 body 的 name

        返回:
            linear_vel  : np.ndarray(3,)   # [vx, vy, vz]  (m/s)
            angular_vel : np.ndarray(3,)   # [wx, wy, wz]  (rad/s)

        异常:
            ValueError : 如果找不到该 body
        """
        # 1. 取得 body id
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的 body")

        # 2. 提取速度
        # ------- MuJoCo 兼容写法 -------
        # 新版(≥2.3.0, 3.x) 提供 xvelp / xangvel
        if hasattr(self.data, "xvelp") and hasattr(self.data, "xangvel"):
            linear_vel  = np.array(self.data.xvelp[body_id])   # 3×1
            angular_vel = np.array(self.data.xangvel[body_id]) # 3×1
        # 旧版可用 cvel（前3维角速度，后3维线速度，均为 body frame）
        else:
            cvel = np.array(self.data.cvel[body_id])           # 6×1
            angular_vel = cvel[:3]
            linear_vel  = cvel[3:]

        return linear_vel, angular_vel
    
    def get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 {site_name} 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos)        # shape (3,)

        # 方向：MuJoCo 已存成9元素向量，无需reshape
        xmat = np.array(self.data.site(site_id).xmat)            # shape (9,)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)                    # [w, x, y, z]

        return position, quaternion

    def mujoco_cam_to_gs(self, cam, scn):
        """
        把 MuJoCo 的 mjvCamera → GSRenderer 需要的 (trans, quat_xyzw)
        GSRenderer 采用 OpenGL 相机系：x→右, y→上, z→前(-z 朝 lookat)
        """
        # -------- 1. 用 mjv_cameraInModel 取位置、forward、up ----------
        pos      = np.zeros(3)   # 相机位置 (世界系)
        forward  = np.zeros(3)   # 从相机指向 look-at
        up_world = np.zeros(3)   # 相机竖直向量
        mujoco.mjv_cameraInModel(pos, forward, up_world, scn)

        # -------- 2. 构造相机自身的三个正交轴 --------------------------
        z_cam = -forward / np.linalg.norm(forward)          # (朝外) OpenGL 相机 –Z
        up_world /= np.linalg.norm(up_world)

        # “右手系”：x = y × z
        x_cam = np.cross(up_world, z_cam)
        x_cam /= np.linalg.norm(x_cam)

        # 重新算严格正交的 y_cam
        y_cam = np.cross(z_cam, x_cam)

        # -------- 3. 得到 R_cw （列向量 = 相机轴在世界系坐标）-----------
        # SciPy 的 Rotation.from_matrix 期望列向量形式
        R_cw = np.stack([x_cam, y_cam, z_cam], axis=1)    # 3×3

        # -------- 4. 四元数：SciPy 返回 xyzw 顺序 ----------------------
        quat_xyzw = Rotation.from_matrix(R_cw).as_quat()  # xyzw

        return pos, quat_xyzw


    def get_img(self):
        rgb_imgs = []
        depth_imgs = []
        rgb_dict = {}
        depth_dict = {}
        start_time = time.perf_counter()
        for cam_id in range(self.model.ncam):
            cam_name = self.model.camera(cam_id).name
            # print(f"Camera ID: {cam_id}, Name: {cam_name}")
            cam_pos = self.data.cam_xpos[cam_id]
            cam_rot = self.data.cam_xmat[cam_id].reshape((3, 3))
            cam_quat = Rotation.from_matrix(cam_rot).as_quat()

            # 设置gs相机参数，渲染图像
            self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)
            self.gs_renderer.set_camera_pose(cam_pos, cam_quat)
            rgb_img, depth_img = self.gs_renderer.render()
            rgb_imgs.append(rgb_img)
            depth_imgs.append(depth_img)
            rgb_dict[cam_name] = rgb_img
            depth_dict[cam_name] = depth_img

        end_time = time.perf_counter()
        # print(f"delta time : {end_time - start_time}")


        # # 获取当前原相机视角的图像
        # self._cam.lookat[:]   = self.handle.cam.lookat
        # self._cam.distance    = self.handle.cam.distance
        # self._cam.azimuth     = self.handle.cam.azimuth
        # self._cam.elevation   = self.handle.cam.elevation
        # self._cam.type        = self.handle.cam.type

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            mujoco.MjvPerturb(),
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scn
        )
        pos, quat_xyzw = self.mujoco_cam_to_gs(
            self._cam,  # mjvCamera
            self._scn   # mjvScene
        )
        self.gs_renderer.set_camera_fovy(self.rgb_fovy * np.pi / 180.)
        self.gs_renderer.set_camera_pose(pos, quat_xyzw)

        free_rgb_img, free_depth_img = self.gs_renderer.render()
        # free_rgb_img = None
        # free_depth_img = None
        return rgb_imgs, depth_imgs, free_rgb_img, free_depth_img,rgb_dict, depth_dict


    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """
        设置模拟器中的状态（关节位置和速度），并同步 forward。
        """
        assert qpos.shape == (self.model.nq,), f"Expected qpos shape ({self.model.nq},), got {qpos.shape}"
        assert qvel.shape == (self.model.nv,), f"Expected qvel shape ({self.model.nv},), got {qvel.shape}"

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)

        if self.model.na == 0 and hasattr(self.data, "act"):
            # 如果没有 actuator，清空 act，否则可能会引发非法内存读写
            self.data.act[:] = 0

        mujoco.mj_forward(self.model, self.data)


    def update_gs_scene(self):
        for name in self.robot_link_list:
            trans, quat_wxyz = self.get_body_pose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        for name in self.obj_list:
            trans, quat_wxyz = self.get_body_pose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        # for name in self.geom_robot_list:
        #     trans, quat_wxyz = self.get_geom_pose(name)

        #     # TODO
        #     # 转换为 scipy 的四元数格式 [x, y, z, w]
        #     quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        #     r_orig = Rotation.from_quat(quat_xyzw)

        #     # 绕 Z 顺时针旋转 90°，也就是 -90°
        #     r_z_neg90 = Rotation.from_euler('x', -90, degrees=True)

        #     # 合成旋转：先原始旋转，再绕 Z 轴旋转
        #     r_new = r_z_neg90 * r_orig

        #     # 转换回 MuJoCo 使用的四元数格式 [w, x, y, z]
        #     quat_new_xyzw = r_new.as_quat()
        #     quat_new_wxyz = [quat_new_xyzw[3], quat_new_xyzw[0], quat_new_xyzw[1], quat_new_xyzw[2]]
        #     quat_new_wxyz = np.array(quat_new_wxyz)

        #     self.gs_renderer.set_obj_pose(name, trans, quat_new_wxyz)
        

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

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    
    def calculate_target_joint6_pos_ori(
        self,
        target_get_ee_pos,
        target_ee_quat_wxyz
    ):
        """
        计算末端夹爪到关节 6 的变换
        
        参数:
            target_get_ee_pos:                     末端夹爪在世界系下的期望位置 (x, y, z)
            target_ee_quat_wxyz:                   末端夹爪在世界系下的期望姿态 (w, x, y, z)
        return:
            link6 pos in world frame & quat (w,x,y,z) trnafered from ee pos
                {'position': (x, y, z), 'quaternion_wxyz': (w, x, y, z)}
        """

        T_link_ee = np.zeros((4,4))
      
        ee_quat_xyzw = np.roll(target_ee_quat_wxyz, -1)

        rot_ee = Rotation.from_quat(ee_quat_xyzw).as_matrix()
       

        T_link_ee[:3,:3] = Rotation.from_matrix(rot_ee).as_matrix()
        T_link_ee[:3,3] = target_get_ee_pos 

        T_from_ee_to_link6 = np.array([[0, 0, 1, -0.085], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        link6_transfered = T_link_ee @ T_from_ee_to_link6

        link6_pos_transfered = np.array(link6_transfered[:3,3])
        link6_xyzw_transfered = Rotation.from_matrix(np.array(link6_transfered[:3,:3])).as_quat()
        link6_wxyz_transfered = np.array([link6_xyzw_transfered[3],link6_xyzw_transfered[0],link6_xyzw_transfered[1],link6_xyzw_transfered[2]])
       
        return link6_pos_transfered, link6_wxyz_transfered
    
    def world_to_local_pose(self, world_obj_pos, world_obj_quat, local_origin_pos, local_origin_quat):
        """
        将世界坐标系下的物体位姿转换到以点a为原点的局部坐标系下的位姿
        
        参数:
            world_obj_pos:       物体在世界坐标系下的位置 [x, y, z]
            world_obj_quat:      物体在世界坐标系下的旋转四元数 [w, x, y, z]
            local_origin_pos:    局部坐标系原点a在世界坐标系下的位置 [x, y, z]
            local_origin_quat:   局部坐标系原点a在世界坐标系下的旋转四元数 [w, x, y, z]
            
        返回:
            local_obj_pos:       物体在局部坐标系下的位置 [x, y, z]
            local_obj_quat:      物体在局部坐标系下的旋转四元数 [w, x, y, z]
        """
        # 计算局部坐标系到世界坐标系的变换矩阵
        R_world_to_local = Rotation.from_quat(local_origin_quat[[1, 2, 3, 0]]).inv()  # 注意 wxyz -> xyzw
        
        # 转换位置
        local_obj_pos = R_world_to_local.apply(world_obj_pos - local_origin_pos)
        
        # 转换旋转 (四元数乘法 q_local = q_a^-1 * q_world)
        rot_obj_world = Rotation.from_quat(world_obj_quat[[1, 2, 3, 0]])
        rot_local = R_world_to_local * rot_obj_world
        local_obj_quat = rot_local.as_quat()[[3, 0, 1, 2]]  # xyzw -> wxyz
        return local_obj_pos, local_obj_quat
    
    def calc_arm_rrt_cubic_traj(
        self,
        cur_joints_state,
        target_joints_state
    ):
        """
        计算机械臂在给定目标关节角度下的运动轨迹
        
        参数:
            cur_joints_state:         当前机械臂的 6 关节状态
            target_joints_state:      机械臂目标 6 关节
        返回:
            path: 轨迹
        """
        q_start = cur_joints_state
        q_goal = target_joints_state

        # Search for a path
        options = RRTPlannerOptions(
            max_step_size=0.05,
            max_connection_dist=5.0,
            rrt_connect=False,
            bidirectional_rrt=True,
            rrt_star=True,
            max_rewire_dist=5.0,
            max_planning_time=20.0,
            fast_return=True,
            goal_biasing_probability=0.15,
            collision_distance_padding=0.01,
        )
        print("")
        print(f"Planning a path...")
        planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
        q_path = planner.plan(q_start, q_goal)
        if len(q_path) > 0:
            print(f"Got a path with {len(q_path)} waypoints")
        else:
            print("Failed to plan.")

        # Perform trajectory optimization.
        dt = 0.025
        options = CubicTrajectoryOptimizationOptions(
            num_waypoints=len(q_path),
            samples_per_segment=7,
            min_segment_time=0.5,
            max_segment_time=10.0,
            min_vel=-1.5,
            max_vel=1.5,
            min_accel=-0.75,
            max_accel=0.75,
            min_jerk=-1.0,
            max_jerk=1.0,
            max_planning_time=30.0,
            check_collisions=True,
            min_collision_dist=self.distance_padding,
            collision_influence_dist=0.05,
            collision_avoidance_cost_weight=0.0,
            collision_link_list=[
                "ground_plane",
                "link6",
            ],
        )
        print("Optimizing the path...")
        optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
        traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

        if traj is not None:
            print("Trajectory optimization successful")
            traj_gen = traj.generate(dt)

        if traj is not None:
            return traj_gen[1]
        else:
            return None
    
    def calc_arm_plan_path(
        self,
        target_pos_world,
        target_pos_ori_world,
        arm_base_pos_world,
        arm_base_ori_world,
        cur_joints_state
    ):
        """
        计算机械臂到世界坐标系下的某一点的运动轨迹
        
        参数:
            target_pos_world:         世界坐标系下的目标位置 (xyz)
            target_pos_ori_world:     世界坐标系下的目标姿态 (wxyz)
            arm_base_pos_world:       世界坐标系下的机械臂基座的位置 (xyz)
            arm_base_ori_world:       世界坐标系下的机械臂基座的姿态 (wxyz)
            cur_joints_state:         当前机械臂的 6 关节状态
        返回:
            path:                     期望轨迹
            target_joints_state:      期望轨迹的最后 6 关节位置
        """

        ## Step 1 计算 link 6 的目标位姿 世界坐标系 !!!
        target_ee_pos = target_pos_world
        target_ee_ori = target_pos_ori_world
        target_link6_pos, target_link6_wxyz = self.calculate_target_joint6_pos_ori(target_ee_pos, target_ee_ori)
        ## Step 2 计算在 arm base 坐标系下 link 6 的目标位姿
        arm_base_pos = arm_base_pos_world
        arm_base_ori = arm_base_ori_world
        local_link6_pos, local_link6_quat_wxyz = self.world_to_local_pose(target_link6_pos, target_link6_wxyz, arm_base_pos, arm_base_ori)
        ## Step 3 调用 ik 求解
        # 目标抓取位置
        target_position = local_link6_pos
        # 目标抓取姿态
        # 假设 local_link6_quat_wxyz 是 [w, x, y, z]
        quat_xyzw = [local_link6_quat_wxyz[1], local_link6_quat_wxyz[2], local_link6_quat_wxyz[3], local_link6_quat_wxyz[0]]

        # 转成 Rotation 对象
        rotation_obj = Rotation.from_quat(quat_xyzw)

        # 转为欧拉角（单位是弧度）
        target_orientation_euler = rotation_obj.as_euler('xyz', degrees=False)

        # 再转成旋转矩阵
        target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]
        # 计算逆运动学解
        target_joint_angles = self.my_chain.inverse_kinematics(target_position, target_orientation, "all")
        target_joints_state = target_joint_angles[1:]

        path = self.calc_arm_rrt_cubic_traj(cur_joints_state, target_joints_state)
        return path, target_joints_state
    
    def normalize_depth(self, depth_img, max_depth=5.0):
        # 裁剪深度值
        depth_img_clipped = np.clip(depth_img, 0, max_depth)
        # 反转归一化：近距离对应高值255，远距离对应低值0
        depth_normalized = ((max_depth - depth_img_clipped) / max_depth * 255).astype(np.uint8)
        return depth_normalized
    
    def get_localization(self):
        lidar_name = "lidar_site"
        lidar_pos, lidar_wxyz = self.get_site_pos_ori(lidar_name)
        return lidar_pos, lidar_wxyz

    
    def run_before(self):
        

        self.init_state = self.data.qpos.copy()

        while True:            
            q_start = self.random_valid_state()
            q_start = np.zeros(6)
            if q_start is None:
                raise RuntimeError("随机采样始终失败，无法规划。请检查碰撞对或关节范围设置。")
            
            

            q_goal = self.random_valid_state()
            print(f"q_start type : {type(q_start)}")
            print(f"q_start : {q_start}")
            print(f"q_goal : {q_goal}")

            # Search for a path
            options = RRTPlannerOptions(
                max_step_size=0.05,
                max_connection_dist=5.0,
                rrt_connect=False,
                bidirectional_rrt=True,
                rrt_star=True,
                max_rewire_dist=5.0,
                max_planning_time=20.0,
                fast_return=True,
                goal_biasing_probability=0.15,
                collision_distance_padding=0.01,
            )
            print("")
            print(f"Planning a path...")
            planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
            q_path = planner.plan(q_start, q_goal)
            if len(q_path) > 0:
                print(f"Got a path with {len(q_path)} waypoints")
            else:
                print("Failed to plan.")

            # Perform trajectory optimization.
            dt = 0.025
            options = CubicTrajectoryOptimizationOptions(
                num_waypoints=len(q_path),
                samples_per_segment=7,
                min_segment_time=0.5,
                max_segment_time=10.0,
                min_vel=-1.5,
                max_vel=1.5,
                min_accel=-0.75,
                max_accel=0.75,
                min_jerk=-1.0,
                max_jerk=1.0,
                max_planning_time=30.0,
                check_collisions=True,
                min_collision_dist=self.distance_padding,
                collision_influence_dist=0.05,
                collision_avoidance_cost_weight=0.0,
                collision_link_list=[
                    # "obstacle_box_1",
                    # "obstacle_box_2",
                    # "obstacle_sphere_1",
                    # "obstacle_sphere_2",
                    "ground_plane",
                    "link6",
                ],
            )
            print("Optimizing the path...")
            optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
            traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

            if traj is None:
                print("Retrying with all the RRT waypoints...")
                traj = optimizer.plan(q_path, init_path=q_path)

            if traj is not None:
                print("Trajectory optimization successful")
                traj_gen = traj.generate(dt)
                self.q_vec = traj_gen[1]
                print(f"path has {self.q_vec.shape[1]} points")
                self.tforms = extract_cartesian_poses(self.model_roboplan, "link6", self.q_vec.T)
                # 提取位置信息
                positions = []
                print(self.tforms[0].translation)
                print(self.tforms[0].rotation)
                self.handle.user_scn.ngeom = 0
                i = 0
                print(f"")
                for i, tform in enumerate(self.tforms):
                    if i % 2 == 0:
                        continue
                    position = tform.translation
                    rotation_matrix = tform.rotation
                    mujoco.mjv_initGeom(
                        self.handle.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.005, 0, 0],
                        pos=np.array([tform.translation[0], tform.translation[1], tform.translation[2]]),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1])
                    )
                    i += 1
                self.handle.user_scn.ngeom = i
                print(f"Added {i} spheres to the scene.")
                for tform in self.tforms:
                    position = tform.translation
                    positions.append(position)

                positions = np.array(positions)

                # 创建 3D 图形
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # 绘制位置轨迹
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o')

                # 绘制姿态
                for i, tform in enumerate(self.tforms):
                    position = tform.translation
                    rotation_matrix = tform.rotation
                    # 提取坐标轴方向的向量
                    x_axis = rotation_matrix[:, 0]
                    y_axis = rotation_matrix[:, 1]
                    z_axis = rotation_matrix[:, 2]
                    # 绘制坐标轴向量
                    ax.quiver(position[0], position[1], position[2],
                            x_axis[0], x_axis[1], x_axis[2], color='r', length=0.01)
                    ax.quiver(position[0], position[1], position[2],
                            y_axis[0], y_axis[1], y_axis[2], color='g', length=0.01)
                    ax.quiver(position[0], position[1], position[2],
                            z_axis[0], z_axis[1], z_axis[2], color='b', length=0.01)

                # 设置坐标轴标签
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # 显示图形
                plt.show(block=False)
                plt.pause(0.001)
                break
        
        self.index = 0
        

    def random_valid_state(self):

        res =  get_random_collision_free_state(
            self.model_roboplan, self.collision_model, distance_padding=0.01
        )
        print("collisionPairs after SRDF :", len(self.collision_model.collisionPairs))
        return res
    
    def show_gs_render(self):
        self.update_gs_scene()
        gs_imgs, depth_gs_imgs, gs_img, depth_gs_img,gs_imgs_dict,_ = self.get_img()            # RGB图

        for obs_, val_ in gs_imgs_dict.items():
            if isinstance(val_, float):
                rr.log(f"observation.{obs_}", rr.Scalar(val_))
                # print(f"{obs_} shape: {val_.shape}, dtype: {val_.dtype}")

            elif isinstance(val_, np.ndarray):
                rr.log(f"observation.{obs_}", rr.Image(val_), static=True)
                # print(f"{obs_} shape: {val_.shape}, dtype: {val_.dtype}")

        # # --- 主相机：RGB + 深度 ---
        # # RGB图像
        # gs_img_bgr = cv2.cvtColor(gs_img, cv2.COLOR_RGB2BGR)
        # # 深度图 → 彩色图
        # depth_img_color = cv2.applyColorMap(self.normalize_depth(depth_gs_img), cv2.COLORMAP_JET)
        
        # # 拼接主相机左右图像 (RGB + Depth)
        # main_concat = np.hstack((gs_img_bgr, depth_img_color))
        
        # # --- 自定义相机数组：RGB + 深度，按行拼接 ---
        # if len(gs_imgs) > 0 and len(depth_gs_imgs) == len(gs_imgs):
        #     # RGB图像转 BGR
        #     gs_imgs_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gs_imgs]
        #     rgb_row = np.hstack(gs_imgs_bgr)
            
        #     # 深度图转彩色图
        #     depth_imgs_color = [cv2.applyColorMap(self.normalize_depth(img), cv2.COLORMAP_JET) for img in depth_gs_imgs]
        #     depth_row = np.hstack(depth_imgs_color)
            
        #     # 上下拼接（RGB 在上，深度在下）
        #     custom_concat = np.vstack((rgb_row, depth_row))
            
        #     # 调整自定义相机图像大小（缩小为原图的1/4）
        #     scale_factor = 0.25
        #     small_custom = cv2.resize(custom_concat, 
        #                             (int(custom_concat.shape[1] * scale_factor), 
        #                             int(custom_concat.shape[0] * scale_factor)))
            
        #     # 将小图嵌入到主相机窗口的右上角
        #     h, w = small_custom.shape[:2]
        #     main_h, main_w = main_concat.shape[:2]
            
        #     # 确保嵌入位置不超出主图像范围
        #     margin = 10  # 边距
        #     pos_x = main_w - w - margin  # 右上角X坐标
        #     pos_y = margin               # 右上角Y坐标
            
        #     # 创建一个主图像的副本（避免修改原始数据）
        #     combined = main_concat.copy()
            
        #     # 将小图覆盖到主图上（ROI操作）
        #     combined[pos_y:pos_y+h, pos_x:pos_x+w] = small_custom
            
        #     # 可选：在小图周围画一个边框
        #     cv2.rectangle(combined, (pos_x, pos_y), (pos_x+w, pos_y+h), (0, 255, 0), 2)
        # else:
        #     combined = main_concat
        
        # # 显示组合窗口
        # cv2.namedWindow("GS Main Camera with PIP", cv2.WINDOW_NORMAL)
        # cv2.startWindowThread()
        # cv2.imshow("GS Main Camera with PIP", combined)
        # cv2.waitKey(1)
        

    def runFunc(self):
        pass