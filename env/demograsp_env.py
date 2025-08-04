from base_env import *
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
import copy
import pygpg
from mujoco import MjModel, MjData, mjtObj
from piper_sdk import C_PiperInterface
import json



import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# class SimpleGraspSampler:
#     def __init__(self,
#                  base_pos: np.ndarray,
#                  base_ori_wxyz: np.ndarray,
#                  num_samples: int = 10,
#                  approach_dist: float = 0.01,
#                  min_dot: float = 0.3,
#                  max_yaw_deg: float = 60.0):
#         """
#         base_pos:       机械臂基座在世界系下的位置 (3,)
#         base_ori_wxyz:  机械臂基座在世界系下的姿态 [w, x, y, z]
#         num_samples:    随机采样的点数量
#         approach_dist:  抓取预留的后退距离
#         min_dot:        法向进给与基座连线方向点积阈值（>min_dot 保证不会背对）
#         max_yaw_deg:    在水平面上进给方向与基座前向最大偏离角度（度）
#         """
#         self.base_pos = np.array(base_pos, dtype=float)
#         self.N = num_samples
#         self.d = approach_dist
#         self.min_dot = min_dot
#         r_base = R.from_quat([base_ori_wxyz[1],
#                               base_ori_wxyz[2],
#                               base_ori_wxyz[3],
#                               base_ori_wxyz[0]])
#         self.f_base = r_base.apply([1, 0, 0])  
#         self.yaw_cos_thresh = np.cos(np.deg2rad(max_yaw_deg))

#     def sample_from_ply(self, ply_path: str, T_aw: np.ndarray) -> list[np.ndarray]:
#         """
#         从给定的 PLY 文件中读取点云，随机生成一批抓取姿态。
#         返回：List[np.ndarray]，每个元素是 4×4 世界系齐次矩阵
#         """
#         # 1. 载入并估计法向
#         pcd = o3d.io.read_point_cloud(ply_path)
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
#         )
#         pcd.normalize_normals()

#         pts_local = np.asarray(pcd.points)   # (M,3)
#         nms_local = np.asarray(pcd.normals)  # (M,3)

#         # 2. 随机选 N 个点
#         idxs = np.random.choice(len(pts_local), size=self.N, replace=False)
#         R_aw = T_aw[:3, :3]
#         t_aw = T_aw[:3,  3]
#         poses = []

#         for i in idxs:
#             p_l = pts_local[i]
#             n_l = nms_local[i]
#             p = R_aw @ p_l + t_aw
#             n = R_aw @ n_l

#             # approach 方向 = -法向
            

#             # 过滤掉背对基座的点
#             v = p -self.base_pos
#             v_norm = v / np.linalg.norm(v)
#             # if np.dot(n, v) < 0:
#             #     n = -n
#             z = -n / np.linalg.norm(n)

#             # 去掉背对
          
#             if np.dot(v_norm, z) < self.min_dot:
#                 continue
#             z_xy = np.array([z[0], z[1], 0.0])
#             norm = np.linalg.norm(z_xy)
#             if norm < 1e-3:
#                 continue
#             z_xy /= norm
#             if self.f_base.dot(z_xy) < self.yaw_cos_thresh:
#                 continue

#             # 构造右手系：先定 z，再找任意正交向量 tmp 构 y，再叉乘得 x
#             tmp = np.array([1.0, 0.0, 0.0])
#             if abs(tmp.dot(z)) > 0.9:
#                 tmp = np.array([0.0, 1.0, 0.0])
#             y = np.cross(z, tmp)
#             y /= np.linalg.norm(y)
#             x = np.cross(y, z)

#             # 3. 拼装 4×4 齐次矩阵
#             T = np.eye(4, dtype=float)
#             T[:3, 0] = z
#             T[:3, 1] = x
#             T[:3, 2] = y
#             # 抓取点从表面向外后退 d
#             T[:3, 3] = p - z * self.d

#             poses.append(T)

#         return poses

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class GPDGraspSampler:
    def __init__(self,
                 num_samples=1000,
                 num_width=5,
                 num_angles=8,
                 gripper_depth=0.06,
                 gripper_width=0.08,
                 finger_width=0.01,
                 rball=0.02):
        """
        num_samples: 随机采点数量 N  
        num_width:   侧向偏移采样点数 |Y|  
        num_angles:  旋转角度采样数 |Φ|  
        gripper_depth:  手指闭合行程深度（m）  
        gripper_width:  手指对开最大宽度（m）  
        finger_width:   单指宽度（m）  
        rball:        Darboux 参考系邻域半径 (m)  
        """
        self.N = num_samples
        self.W = num_width
        self.A = num_angles
        self.depth = gripper_depth
        self.width = gripper_width
        self.finger = finger_width
        self.rball = rball
        # 侧向偏移范围：[-w/2, ..., +w/2]
        self.Y = np.linspace(-self.width/2, self.width/2, self.W)
        # 角度采样
        self.PHI = np.linspace(0, 2*np.pi, self.A, endpoint=False)

    def _compute_darboux_frame(self, pcd, idx):
        """在第 idx 个点邻域里估计Darbox坐标系"""
        pts = np.asarray(pcd.points)
        nms = np.asarray(pcd.normals)
        p = pts[idx]
        # 找邻域内所有点
        dists = np.linalg.norm(pts - p, axis=1)
        neigh = np.where(dists < self.rball)[0]
        # 构造 M = Σ n(q)n(q)^T
        M = np.zeros((3,3))
        for j in neigh:
            nj = nms[j].reshape(3,1)
            M += nj @ nj.T
        # 特征分解
        w, V = np.linalg.eigh(M)
        # v3: 最小特征向量（平滑法线），v1,v2为主曲率方向
        idxs = np.argsort(w)
        v3 = V[:, idxs[0]]
        v1 = V[:, idxs[2]]
        v2 = V[:, idxs[1]]
        # 保证 v3 指向外侧（与原始法线同向）
        if np.dot(v3, nms[idx]) < 0:
            v3 = -v3
        return np.column_stack([v1, v2, v3])  # 注意：后面我们把 z 轴当作 approach

    def sample(self, world_pcd: o3d.geometry.PointCloud):
        """
        输入：估计完法向量的世界坐标系点云
        输出：List[np.ndarray]，每个 (4×4) 世界系 抓取变换矩阵
        """
        # 1) 预处理：下采样 + 法线估计
        pcd = world_pcd.voxel_down_sample(voxel_size=0.002)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pts = np.asarray(pcd.points)
        nms = np.asarray(pcd.normals)

        N = len(pts)
        # 2) 随机选 N_sample 个点
        idxs = np.random.choice(N, size=min(self.N, N), replace=False)
        grasps = []
        # 构建 KDTree 以便快速查询碰撞
        kdt = o3d.geometry.KDTreeFlann(pcd)

        for i in idxs:
            p = pts[i]
            # 3) Darboux 局部参考系 F(p)
            R_p = self._compute_darboux_frame(pcd, i)
            T_p = np.eye(4)
            T_p[:3,:3] = R_p
            T_p[:3, 3] = p

            # 4) 网格 (y, φ) 遍历
            for y in self.Y:
                for phi in self.PHI:
                    # 4.1) 在局部系内构造偏移 + 旋转
                    T_offset = np.eye(4)
                    # 绕 z 轴旋转 φ
                    T_offset[:3,:3] = R.from_euler('z', phi).as_matrix()
                    # 侧向平移 y（对应 F(p) 中的 y 轴）
                    T_offset[:3,3] = np.array([0.0, y, 0.0])

                    # 4.2) 初步位姿
                    T = T_p @ T_offset

                    # 5) “推”到第一个无碰撞位置：不断增加 x 方向偏移
                    #    直到掌心（或手指底座）模块不与点云重叠
                    #    这里简化：最多推进 depth 步长，
                    #    实际可细化成二分或步长搜索
                    for step in np.linspace(0, self.depth, 5):
                        T_try = T.copy()
                        T_try[:3,3] -= step * T_try[:3,2]  # 沿 local z 轴（法线方向）后退
                        # 检测掌心体积是否与点云相交
                        # 用一个小球近似掌心位置
                        center = T_try[:3,3]
                        [_, idx_n, _] = kdt.search_radius_vector_3d(center, self.finger)
                        if len(idx_n) == 0:
                            # 没碰撞，接受此步
                            T = T_try
                            break
                    else:
                        # 始终碰撞，丢弃
                        continue

                    # 6) 闭合区域检测：在两个指面之间是否有点
                    # 指面平面方程：点到两平面距离 < finger_width/2
                    # 简化：计算 candidate 点云到掌心的相对坐标
                    rel = (pts - T[:3,3]) @ T[:3,:3]  # 投影到 local frame
                    cond = (np.abs(rel[:,1]) < (self.finger/2)) & \
                           (rel[:,0] > 0) & (rel[:,0] < self.depth)
                    if np.any(cond):
                        grasps.append(T)

        return grasps



class DemoGrasp(BaseEnv):
    def __init__(self, path, cfg):
        super().__init__(path, cfg)
        self.path = path
        self.render_lock = threading.Lock()
        for i in range(1,9):
            self.add_gaussian_model(f"link{i}",f"/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link{i}_rot.ply")
        self.add_gaussian_model(
            "apple", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply"
        )
        self.add_gaussian_model(
            "banana", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/banana/banana_res.ply")
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()

        self.piper = C_PiperInterface("can0")
        self.piper.ConnectPort()
        self.factor = 57324.840764
        self.targets = [o for o in cfg["obj_list"] if o not in ["desk"]]
        # body id 缓存
        self.target_body_ids = {
            name: mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, name)
            for name in self.targets
        }
        self.gripper_body_id = mujoco.mj_name2id(
            self.model, mjtObj.mjOBJ_BODY, "link7")
        # 存放已录制的结果
        self.recorded = {}
        self.out_file = "recorded_grasps.json"
        self.close_thresh = 0.005
        # self.sampler = SimpleGraspSampler()

    def run_before(self):
        self.tele_action = []
        self.recorded.clear()
        print("[DemoGrasp] 开始遥操，夹爪闭合并接触时将自动保存抓取位姿 →",
              self.out_file)


    def is_gripper_closed(self):
        # joint7,qpos[6] & joint8,qpos[7]
        return (self.data.qpos[6] < self.close_thresh 
                and abs(self.data.qpos[7]) < self.close_thresh)
    
    def visualize_fork_gripper(self, T: np.ndarray, pause: float = 0.5):
        """
        用 fork_gripper 这个 6DOF 自由关节去可视化给定的 4x4 世界系 抓取齐次矩阵 T
        pause: 在每帧之间暂停的秒数
        """
        # 1) body/joint 找地址
        body_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "fork_gripper")
        jnt_id  = self.model.body_jntadr[body_id]
        qpos_adr = self.model.jnt_qposadr[jnt_id]

        # 2) 提取平移和旋转（转四元数 wxyz）
        pos = T[:3, 3]
        mat = T[:3, :3]
        quat_xyzw = R.from_matrix(mat).as_quat()       # SciPy → [x,y,z,w]
        quat_wxyz = np.roll(quat_xyzw, 1)             # 转成 MuJoCo [w,x,y,z]

        # 3) 写入 qpos，并清零速度防止弹飞
        self.data.qpos[qpos_adr     : qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
        self.data.qvel[:] = 0
        if hasattr(self.data, "qacc"):
            self.data.qacc[:] = 0

        # 4) 同步仿真 + GS 渲染
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        self.update_gs_scene()

        # 5) 短暂暂停，方便肉眼观测
        time.sleep(pause)

    def render_loop(self):
        
        while True:
            self.show_gs_render()


    def run_func(self):
        base_pos, base_ori = self.get_body_pose("base_link")
        pos_o, quat_o = self.get_body_pose("banana")      # [x,y,z], [w,x,y,z]
        R_aw = R.from_quat([quat_o[1], quat_o[2], quat_o[3], quat_o[0]]).as_matrix()
        T_aw = np.eye(4)
        T_aw[:3,:3] = R_aw
        T_aw[:3, 3] = pos_o
        sampler = GPDGraspSampler(
            num_samples=1000,    # 随机选点数
            num_width=5,         # 侧移离散点数
            num_angles=8,        # 法线轴上旋转角度数
            gripper_depth=0.06,  # 手指闭合深度
            gripper_width=0.08,  # 最大张开宽度
            finger_width=0.01,   # 单指宽度
            rball=0.02           # Darboux 半径
        )
        candidates = sampler.sample(world_pcd)   # List[np.ndarray(4×4)]
        print(f"生成了 {len(candidates)} 个候选抓取。")
        print(f"为苹果生成 {len(apple_poses)} 个抓取姿态")
        print(apple_poses)
        for T in apple_poses:
            self.visualize_fork_gripper(T, pause=0.5)
        return
        # self.tele_action = []
        # arm_msgs = self.piper.GetArmJointMsgs()
        # gripper_action = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle* (1/(70*1000)) * 0.035
        # for i in range(1, 7):
        #     angle = getattr(arm_msgs.joint_state, f"joint_{i}")
        #     self.tele_action.append(float(angle) * (1/self.factor))
        # # self.tele_action.append(float(gripper_msgs.gripper_state.grippers_angle) )
        
        # # # time.sleep(0.002)
        # # print(self.tele_action)
        # self.data.qpos[:6] = self.tele_action[:6]
        # self.data.qpos[6] = gripper_action
        # self.data.qpos[7] = -gripper_action
        # mujoco.mj_forward(self.model, self.data)
        # self.sync()
      
        # self.update_gs_scene()

        # if self.is_gripper_closed():
        #     for ci in range(self.data.ncon):
        #         c = self.data.contact[ci]
        #         b1 = self.model.geom_bodyid[c.geom1]
        #         b2 = self.model.geom_bodyid[c.geom2]
        #         # 如果跟 gripper 和某个目标物体接触
        #         if self.gripper_body_id in (b1, b2):
        #             for name, bid in self.target_body_ids.items():
        #                 if bid in (b1, b2):
        #                     print(f"\n检测到与 `{name}` 接触，抓取姿态如下：")
        #                     # 4) 计算并显示当前世界系 T_gripper
        #                     pos_g, quat_g = self.get_body_pose("fork_gripper")
        #                     T_g = self.pose_to_transform(pos_g, quat_g)
        #                     print(np.array2string(T_g, precision=3, suppress_small=True))

        #                     # 5) 阻塞式用户确认
        #                     resp = input("按 Enter 保存此抓取；按其它键+Enter 丢弃： ")
        #                     if resp == "":
        #                         # 保存相对姿态
        #                         pos_o, quat_o = self.get_body_pose(name)
        #                         T_o = self.pose_to_transform(pos_o, quat_o)
        #                         T_rel = np.linalg.inv(T_o) @ T_g
        #                         self.recorded[name] = T_rel.tolist()
        #                         # 立刻写文件
        #                         with open(self.out_file, "w") as f:
        #                             json.dump(self.recorded, f, indent=2)
        #                         print(f"[保存] `{name}` 的相对抓取姿态已写入 {self.out_file}")
        #                     else:
        #                         print(f"[丢弃] `{name}` 这次演示未保存。")

        #                     # ——— 新增：环境重置逻辑 ———
        #                     # （1）先把仿真环境里所有机械臂关节置零
        #                     self.data.qpos[:6] = np.zeros(6)
        #                     mujoco.mj_forward(self.model, self.data)
        #                     self.sync()
        #                     self.update_gs_scene()

        #                     # （2）提示并等待 5 秒，让用户把真实手臂回到 0
        #                     print("\n>>> 请将遥操作机械臂移动回 0 位，等待 5 秒…")
        #                     time.sleep(5)

        #                     # （3）结束本次 blocking，返回继续下一演示
        #                     print("\n=== 环境已重置，开始下一次演示 ===")
        #                     return

        



    
if __name__ == "__main__":
    cfg = {
        "is_have_moving_base": False,
        "moving_base": {
            "wheel_radius": 0.05,
            "half_wheelbase": 0.2
        },
        "is_have_arm": True,
        "is_use_gs_render": True,
        "episode_len": 100,
        "is_save_record_data": False,
        "camera_names": ["3rd_camera", "wrist_cam"],
        "robot_link_list": ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"],
        "env_name": "MyMobileRobotEnv",
        "obj_list": ["desk","apple","banana"]
    }
    test = DemoGrasp("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/scene.xml", cfg)
    test.run_loop()
