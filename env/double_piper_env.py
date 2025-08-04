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

class MobileAiRobotEnv(BaseEnv):
    def __init__(self, path, cfg):
        super().__init__(path, cfg)
        self.path = path

    def run_before(self):
        
        ### 读取苹果 ply
        apple_pcd_local = o3d.io.read_point_cloud("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/point_cloud_asserts/object/apple/apple.ply")
        apple_mesh = o3d.io.read_triangle_mesh("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/point_cloud_asserts/object/apple/apple.ply")
        apple_mesh.compute_vertex_normals()
        ### 读取苹果当前世界坐标系下的位姿
        apple_body_name = "apple"
        apple_pos, apple_quat_wxyz = self.get_body_pose(apple_body_name)
        R_aw = R.from_quat([apple_quat_wxyz[1], apple_quat_wxyz[2], apple_quat_wxyz[3], apple_quat_wxyz[0]]).as_matrix()
        T_aw = np.eye(4);  T_aw[:3,:3] = R_aw;  T_aw[:3,3] = apple_pos
        apple_pcd_world = copy.deepcopy(apple_pcd_local).transform(T_aw)

        T_apple_world = np.eye(4)
        T_apple_world[:3,:3]=R_aw
        T_apple_world[:3,3]=apple_pos

        apple_mesh_transformed = apple_mesh.transform(T_apple_world)

        # 创建一个坐标轴（默认长度为1）
        apple_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)  # 你可以调节size
        apple_frame.transform(T_apple_world)  # 把它放在苹果的位置

 


        ### 得到相机位置
        camera_site_name = "camera_site"
        camera_pos, camera_quat_wxyz = self.get_site_pos_ori(camera_site_name)

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
        num_samples = 100
        gripper_config_file = "/home/jiziheng/Music/robot/gs_scene/gs_hs/submodules/pygpg/dependencies/gpg/cfg/gripper_params.cfg"
        pts_cam_std = np.asarray(apple_pcd_camera.points)      # (N,3)
        grasps = pygpg.generate_grasps(pts_cam_std, num_samples,
                                    False,
                                    gripper_config_file)
        

        pose_list = []
        for grasp in grasps:
            pose = np.eye(4)
            pose[:3, 0] = grasp.get_grasp_approach()
            pose[:3, 1] = grasp.get_grasp_binormal()
            pose[:3, 2] = grasp.get_grasp_axis()
            pose[:3, 3] = grasp.get_grasp_bottom()
            pose_list.append(pose)

        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )

        # ---------- 抓取姿态 pose 从相机坐标系转换到世界坐标系 ----------
        pose_list_world = []
        for pose in pose_list:
            pose_world = T_cw @ pose  # T_cw: 相机→世界的变换矩阵
            pose_list_world.append(pose_world)

        # ---------- 过滤掉在苹果 body z 以下的姿态 ----------
        apple_z_world = apple_pos[2]
        filtered_pose_list = []
        for pose in pose_list_world:
            grasp_z = pose[2, 3]  # 抓取位置的 z 值（世界坐标系）
            if grasp_z >= apple_z_world:
                filtered_pose_list.append(pose)

        # visualize pose grasp
        
        # apple_pcd_world.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色
        # def frame_from_pose(pose, size=0.03):
        #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        #     return frame.transform(pose)
        # print(f"number of grasp: {len(pose_list_world)}")
        # # grasp_frames = [frame_from_pose(pose) for pose in filtered_pose_list[0]]
        # print(pose_list_world)
        # grasp_frames = [frame_from_pose( pose_list_world[0])]

        # o3d.visualization.draw_geometries(
        #     [apple_pcd_world,apple_frame, *grasp_frames],
        #     window_name="Grasp + Camera Pose",
        #     width=960, height=720
        # )
        # vis

        self.qpos_list = []
        for pose in filtered_pose_list:
            # 提取平移部分
            position = pose[:3, 3]  # [x, y, z]

            # 提取旋转部分并转为四元数（wxyz）
            rotation_matrix = pose[:3, :3]
            quat_wxyz = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
            quat_wxyz = np.roll(quat_wxyz, 1)  # 转为 [w, x, y, z]

            # 组合成 MuJoCo 的 qpos
            qpos = np.concatenate([position, quat_wxyz])  # [x, y, z, qw, qx, qy, qz]
            self.qpos_list.append(qpos)


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
        qpos = self.qpos_list[0]
        pos = qpos[:3]
        quat_wxyz = qpos[3:]  # [qw, qx, qy, qz]
        tgt_pos = pos
        tgt_ori = quat_wxyz

        base_pos, base_ori = self.get_body_pose("base_link")

        ref_q = np.zeros(6)

        self.q_vec, q_last = self.calc_arm_plan_path(
            tgt_pos, tgt_ori,
            base_pos, base_ori,
            ref_q
        )

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

        body_name = "fork_gripper"
        body_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, body_name)


        # dummy_gripper_pos, dummy_gripper_quat_wxyz = self.get_body_pose("dummy_gripper")
        # link6_pos, link6_quat_wxyz = self.get_body_pose("link6")
        # print("=============pose=============")
        # print(f"dummy_gripper_pos: {dummy_gripper_pos},dummy_gripper_quat_wxyz: {dummy_gripper_quat_wxyz} ")
        # print(f"link6_pos: {link6_pos},link6_quat_wxyz: {link6_quat_wxyz} ")

        # trans_matrix_from_6_to_dummy , _ = self.trans_matrix(dummy_gripper_pos, dummy_gripper_quat_wxyz, link6_pos , link6_quat_wxyz)
        # print(f" trans from 6 to dummy: {trans_matrix_from_6_to_dummy}")

        # qpos索引
        jntadr = self.model.body_jntadr[body_id]
        qpos = self.qpos_list[0]


        pos = qpos[:3]
        quat_wxyz = qpos[3:]  # [qw, qx, qy, qz]

        # 设置位置
        self.data.qpos[jntadr:jntadr+3] = pos

        # 设置旋转四元数
        self.data.qpos[jntadr+3:jntadr+7] = quat_wxyz
        self.index = self.q_vec.shape[1]-1
        self.data.ctrl[:6] = self.q_vec[:6, self.index]
        self.data.ctrl[6] = 1.0
        # self.index += 1
        # if self.index >= self.q_vec.shape[1]:
        #     self.index = 0

        time.sleep(0.01)




if __name__ == "__main__":
    cfg = {
        "is_have_moving_base": True,
        "moving_base": {
            "wheel_radius": 0.05,
            "half_wheelbase": 0.2
        },
        "is_have_arm": True,
        "is_use_gs_render": False,
        "episode_len": 100,
        "is_save_record_data": True,
        "camera_names": ["cam0", "cam1"],
        "robot_link_list": [],
        "env_name": "MyMobileRobotEnv"
    }
    test = MobileAiRobotEnv("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/fix_dual_piper/scene.xml", cfg)
    test.run_loop()