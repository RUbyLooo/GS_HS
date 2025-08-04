import os
os.environ["MUJOCO_GL"] = "egl"
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from typing import List
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
# import faulthandler
# faulthandler.enable()




class MobileAiRobotEnv(BaseEnv):
    def __init__(self, path, cfg):
        super().__init__(path, cfg)
        self.path = path
        # self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        # self.render_thread.start()


    def filter_grasp_poses_world(
        self,
        pose_list_world: List[np.ndarray],
        base_pos: np.ndarray,
        base_ori_wxyz: np.ndarray,
        obj_pos: np.ndarray
    ) -> List[np.ndarray]:
        """
        过滤世界坐标系下的抓取姿态列表，只保留：
        1) 抓取 approach（pose[:3,0]）在 XY 平面投影与 base_link X 轴夹角为锐角
        2) 抓手本地 Z 轴（pose[:3,2]）在世界系中 Z 分量 >= 0（朝上或斜上）

        参数：
        pose_list_world:   List of 4×4 numpy.ndarray 抓取姿态（世界系）
        base_ori_wxyz:     base_link 朝向四元数 [w, x, y, z]

        返回：
        filtered:          满足条件的 world‐frame 抓取姿态列表
        """
        # 1) 计算 base_link 在世界系下的 X 轴 unit vector
        #    base_ori_wxyz 是 [w, x, y, z]
        Rb     = R.from_quat([
            base_ori_wxyz[1],
            base_ori_wxyz[2],
            base_ori_wxyz[3],
            base_ori_wxyz[0]
        ]).as_matrix()
        base_x = Rb[:, 0]  # 世界系下 base_link 本地 X 轴
        base_y = Rb[:,1]   # 本地 Y

        obj_rel = obj_pos - base_pos
        sign_side = np.sign(obj_rel.dot(base_y))  # +1 左侧，-1 右侧

        filtered = []
        for Tw in pose_list_world:
            # approach = 抓手 local X 轴在世界系下的方向
            approach = Tw[:3, 0]
            # z_axis  = 抓手 local Z 轴在世界系下的方向
            z_axis   = Tw[:3, 2]

            # 1) 判断 XY 平面投影后夹角是否锐角
            a_xy = np.array([approach[0], approach[1], 0.0])
            b_xy = np.array([base_x[0],     base_x[1],     0.0])
            if np.dot(a_xy, b_xy) <= 0:
                continue

            # ---- 2) 同侧判断 ----
            # approach 在基座左右侧的标志
            approach_side = np.sign(a_xy.dot(base_y))
            if approach_side != sign_side or sign_side == 0:
                continue

            # # 2) 抓手 Z 轴必须朝上或水平（世界 Z 分量 >= 0）
            # if z_axis[2] < 0:
            #     continue

            filtered.append(Tw)

        return filtered

    def run_before(self):
        

        ### 读取苹果 ply
        apple_pcd_local = o3d.io.read_point_cloud("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/point_cloud_asserts/object/banana/banana.ply")

        ### 读取苹果当前世界坐标系下的位姿
        apple_body_name = "banana"
        apple_pos, apple_quat_wxyz = self.get_body_pose(apple_body_name)
        R_aw = R.from_quat([apple_quat_wxyz[1], apple_quat_wxyz[2], apple_quat_wxyz[3], apple_quat_wxyz[0]]).as_matrix()
        T_aw = np.eye(4);  T_aw[:3,:3] = R_aw;  T_aw[:3,3] = apple_pos
        apple_pcd_world = copy.deepcopy(apple_pcd_local).transform(T_aw)


        ### 得到相机位置
        camera_site_name = "camera_site"
        camera_pos, camera_quat_wxyz = self.get_site_pos_ori(camera_site_name)

        base_pos, base_ori_wxyz = self.get_body_pose("base_link")
        obj_pos,  obj_quat_wxyz = self.get_body_pose("banana")   # 或你要抓的物体

        # x = obj_pos - base_pos
        # x = x / np.linalg.norm(x)
        # up = np.array([0,0,1])
        # # 如果 x 和 up 共线，就换一个 up，比如 [1,0,0]
        # if abs(np.dot(x, up)) > 0.99:
        #     up = np.array([1,0,0])

        # y = np.cross(up, x)
        # y = y / np.linalg.norm(y)

        # z = np.cross(x, y)   # 保证右手
        # z = z / np.linalg.norm(z)

        # R_new = np.stack([x, y, z], axis=1)   # 3×3

        # # 3) 构造 4×4 变换：世界 → 新坐标系
        # T_new = np.eye(4)
        # T_new[:3,:3] = R_new.T   
        # T_new[:3, 3]  = -R_new.T @ base_pos  
        # quat_xyzw = R.from_matrix(R_new).as_quat()         # SciPy 返回 [x,y,z,w]
        # quat_wxyz = np.array([quat_xyzw[3],                 # 转成 [w,x,y,z]
        #                     quat_xyzw[0],
        #                     quat_xyzw[1],
        #                     quat_xyzw[2]])

        # joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_frame_site")
        # qpos_addr = self.model.jnt_qposadr[joint_id]
        # vec7 = np.concatenate([base_pos, quat_wxyz])
        # self.data.qpos[qpos_addr:qpos_addr+7] = vec7
        # mujoco.mj_forward(self.model, self.data)

        # # 4) 转换点云
        # pcd_new = copy.deepcopy(apple_pcd_world).transform(T_new)
        # pts_new = np.asarray(pcd_new.points)

        # 构建 T_cw（相机在世界坐标系下的变换矩阵）
        R_cw = R.from_quat([
            camera_quat_wxyz[1],
            camera_quat_wxyz[2],
            camera_quat_wxyz[3],
            camera_quat_wxyz[0]
        ]).as_matrix()

        T_cw = np.eye(4)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = camera_pos

        # 计算 T_wc^-1，即世界到相机的变换
        T_wc_inv = np.linalg.inv(T_cw)

        # 转换苹果点云到相机坐标系
        apple_pcd_camera = copy.deepcopy(apple_pcd_world).transform(T_wc_inv)

        # ---------- ⑤ 调用 GPD 生成抓取（camera_std 系） ----------
        num_samples = 50
        gripper_config_file = "/home/jiziheng/Music/robot/gs_scene/gs_hs/submodules/pygpg/gripper_params.cfg"
        pts_cam_std = np.asarray(apple_pcd_camera.points)      # (N,3)

      
        grasps = pygpg.generate_grasps(pts_cam_std, num_samples,
                                    True,
                                    gripper_config_file)
        
        self.qpos_list = []
        pose_list = []
        for grasp in grasps:
            pose = np.eye(4)
            # pose[:3, 0] = grasp.get_grasp_approach()
            # pose[:3, 1] = grasp.get_grasp_axis()
            # pose[:3, 2] = -grasp.get_grasp_binormal()
            # pose[:3, 3] = grasp.get_grasp_bottom()
            pose[:3, 0] = grasp.get_grasp_approach()
            pose[:3, 1] = grasp.get_grasp_binormal()
            pose[:3, 2] = grasp.get_grasp_axis()
            pose[:3, 3] = grasp.get_grasp_bottom()
          
            pose_list.append(pose)
          
        # ---------- 抓取姿态 pose 从相机坐标系转换到世界坐标系 ----------
        pose_list_world = []
        for pose in pose_list:
            pose_world = T_cw @ pose  # T_cw: 相机→世界的变换矩阵
            # pose_world =  np.linalg.inv(T_new) @ pose
            pose_list_world.append(pose_world)

        filtered_list = self.filter_grasp_poses_world(pose_list_world, base_pos=base_pos, base_ori_wxyz=base_ori_wxyz, obj_pos=obj_pos)
        print(f"过滤后还有 {len(filtered_list)} 个抓取姿态")

        # # ---------- 过滤掉在苹果 body z 以下的姿态 ----------
        # apple_z_world = apple_pos[2]
        # filtered_pose_list = []
        # for pose in pose_list_world:
        #     grasp_z = pose[2, 3]  # 抓取位置的 z 值（世界坐标系）
        #     if grasp_z >= apple_z_world:
        #         filtered_pose_list.append(pose)

        self.qpos_list = []
        for pose in pose_list_world:
            # 提取平移部分
            position = pose[:3, 3]  # [x, y, z]

            # 提取旋转部分并转为四元数（wxyz）
            rotation_matrix = pose[:3, :3]
            quat_wxyz = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
            quat_wxyz = np.roll(quat_wxyz, 1)  # 转为 [w, x, y, z]

            # 组合成 MuJoCo 的 qpos
            qpos = np.concatenate([position, quat_wxyz])  # [x, y, z, qw, qx, qy, qz]
            self.qpos_list.append(qpos)


        # 目标抓取pose
        base_pos, base_ori = self.get_body_pose("base_link")
        # self.qpos_list = self.filter_grasp_poses(base_pos, base_ori, self.qpos_list)

        ref_q = np.zeros(6)
        print(f"self.qpos_list : {len(self.qpos_list)}")
        self.qpos = self.qpos_list[0]
        # self.qpos = pose_list[0]
        pos = qpos[:3]
        quat_wxyz = qpos[3:]  # [qw, qx, qy, qz]
        tgt_pos = pos
        tgt_ori = quat_wxyz
        self.q_vec, q_last= self.calc_arm_plan_path(
            tgt_pos, tgt_ori,
            base_pos, base_ori,
            ref_q
        )
        
        # world_obj_pos, world_obj_ori = self.local_to_world_pose(self.fk_pose[:3], self.fk_pose[3:7], base_pos, base_ori)
        # self.fk_pose = np.concatenate((world_obj_pos, world_obj_ori), axis=0)

    def trans_matrix(self, pos1, quat_wxyz1, pos2, quat_wxyz2):
        '''
        Para:
            - pos1, quat_wxyz1: 第一个body pose
            - pos2, quat_wxyz2: 第一个body pose
        Return:
            - T_1to2: link1 to link2 transformation(第二个link在第一个link下的坐标)
            - T_2to1: link2 to link1 transformation(第二个link在第一个link下的坐标)
        '''
        T1 = self.pose_to_transform(pos1, quat_wxyz1)
        T2 = self.pose_to_transform(pos2, quat_wxyz2)
        T_1to2 = np.linalg.inv(T1) @ T2
        T_2to1 = np.linalg.inv(T2) @ T1

        return T_1to2, T_2to1


    def pose_to_transform(self, pos, quat_wxyz):
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # wxyz → xyzw
        rot_matrix = R.from_quat(quat_xyzw).as_matrix()  # 3x3
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = pos
        return T


    def run_func(self):
        # 获取该joint的索引（7个自由度：pos + quat）
        fork_gripper_body_name = "fork_gripper"
        fork_gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, fork_gripper_body_name)

        # 获取 joint ID 和 qpos 起始索引
        fork_gripper_joint_id = self.model.body_jntadr[fork_gripper_body_id]
        fork_gripper_qposadr = self.model.jnt_qposadr[fork_gripper_joint_id]
        fork_gripper_pos = self.qpos[:3]
        fork_gripper_quat_wxyz = self.qpos[3:7]

        # 设置位姿
        if fork_gripper_qposadr + 7 <= self.model.nq:
            self.data.qpos[fork_gripper_qposadr     : fork_gripper_qposadr + 3] = fork_gripper_pos
            self.data.qpos[fork_gripper_qposadr + 3 : fork_gripper_qposadr + 7] = fork_gripper_quat_wxyz
        else:
            print("[警告] fork_gripper 的 qpos 索引越界或 joint 设置有误")



        self.index = self.q_vec.shape[1]-1
        self.data.ctrl[:6] = self.q_vec[:6, self.index]
        self.data.ctrl[6] = 1
        # self.index += 1
        # if self.index >= self.q_vec.shape[1]:
        #     self.index = 0

        time.sleep(0.01)




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
        "robot_link_list": [],
        "env_name": "MyMobileRobotEnv",
        "obj_list": ["desk","apple","banana"]
    }
    test = MobileAiRobotEnv("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/scene.xml", cfg)
    # time.sleep(5)
    test.run_loop()