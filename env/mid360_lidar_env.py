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

from se3_controller import *
from motor_mixer import *
import threading

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
import rerun as rr

class Mid360LidarEnv(BaseEnv):
    def __init__(self, path, cfg):
        super().__init__(path, cfg)
        self.path = path

        # ROS 节点和订阅器初始化
        rospy.init_node("mid360_env", anonymous=True)
        rr.init("livox_pointcloud_vis", spawn=True)


        self.localization_pose = {
            "position": np.zeros(3),
            "orientation": np.array([0, 0, 0, 1])  # quaternion wxyz
        }
        rospy.Subscriber("/localization", Odometry, self.localization_callback)
        

        self.global_map = self.load_point_cloud("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/scene/global_map.pcd")
        self.global_map_pub = rospy.Publisher("/map_points", PointCloud2, queue_size=1)

        ### livox lidar 相关
        self.livox_generator = LivoxGenerator("mid360")  # 可选: "avia", "mid40", "mid70", "mid360", "tele"
        self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        print(f"self.rays_theta : {self.rays_theta}, self.rays_phi : {self.rays_phi}")
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        print("site_id =", site_id)
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
        # 创建TF广播者
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pub_lidar = rospy.Publisher('/livox_points', PointCloud2, queue_size=1)

        self.last_lidar_pub_time = rospy.Time.now()
        self.lidar_pub_interval = rospy.Duration(0.1)  # 10Hz = 0.1s

        ### 全局点云相关
        if self.global_map is not None and len(self.global_map) > 0:
            self.global_map_msg = self.xyz_array_to_pointcloud2(self.global_map, frame_id="world")

    def localization_callback(self, msg):
        print(f"new message !")
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.localization_pose["position"] = np.array([pos.x, pos.y, pos.z])
        self.localization_pose["orientation"] = np.array([ori.x, ori.y, ori.z, ori.w])
            

    def broadcast_tf(self, broadcaster, parent_frame, child_frame, translation, rotation, stamp=None):
        """广播TF变换"""
        if stamp is None:
            stamp = rospy.Time.now()
            
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]
        
        broadcaster.sendTransform(t)


    def load_point_cloud(self, path):
        """读取 PCD/PLY -> Nx3 numpy"""
        if not os.path.isfile(path):
            rospy.logerr("Point cloud file not found: %s", path)
            rospy.signal_shutdown("file missing")
            return None
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            rospy.logerr("Loaded point cloud is empty.")
            rospy.signal_shutdown("empty pcd")
            return None
        return np.asarray(pcd.points)   # (N,3)

    def xyz_array_to_pointcloud2(self, points_xyz, frame_id="world"):
        """Nx3 numpy -> PointCloud2"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        fields = [
            PointField("x", 0,  PointField.FLOAT32, 1),
            PointField("y", 4,  PointField.FLOAT32, 1),
            PointField("z", 8,  PointField.FLOAT32, 1),
        ]
        # 每条数据 = 3*4 字节
        cloud_data = points_xyz.astype(np.float32)
        return pc2.create_cloud(header, fields, cloud_data)
    
    def publish_point_cloud(self, publisher, points, frame_id):
        """将点云数据发布为ROS PointCloud2消息"""
        stamp = rospy.Time.now()
            
        # 定义点云字段
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # 添加强度值
        if len(points.shape) == 2:
            # 如果是(N, 3)形状，转换为(3, N)以便处理
            points_transposed = points.T if points.shape[1] == 3 else points
            
            if points_transposed.shape[0] == 3:
                # 添加强度通道
                points_with_intensity = np.vstack([
                    points_transposed, 
                    np.ones(points_transposed.shape[1], dtype=np.float32)
                ])
            else:
                points_with_intensity = points_transposed
        else:
            # 如果点云已经是(3, N)形状
            if points.shape[0] == 3:
                points_with_intensity = np.vstack([
                    points, 
                    np.ones(points.shape[1], dtype=np.float32)
                ])
            else:
                points_with_intensity = points
            
        # 转换为ROS消息格式的点云
        pc_msg = pc2.create_cloud(
            header=rospy.Header(frame_id=frame_id, stamp=stamp),
            fields=fields,
            points=np.transpose(points_with_intensity)  # 转置回(N, 4)格式
        )
        
        publisher.publish(pc_msg)

    def run_before(self):
        pass

    def run_func(self):
        now = rospy.Time.now()
        if now - self.last_lidar_pub_time >= self.lidar_pub_interval:
            self.last_lidar_pub_time = now

            # 设置 lidar_site pose
            pos = self.localization_pose["position"]
            quat = self.localization_pose["orientation"]  # ROS xyzw 四元数
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
            site_body_id = self.model.site_bodyid[site_id]
            mocap_id = self.model.body_mocapid[site_body_id]

            if mocap_id != -1:
                pos = self.localization_pose["position"]
                quat = self.localization_pose["orientation"]  # ROS xyzw
                quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]

                self.data.mocap_pos[mocap_id] = pos
                self.data.mocap_quat[mocap_id] = quat_wxyz
            else:
                print("❌ site 所在 body 不是 mocap 控制体，位置不会生效！")

            # 更新场景并发布雷达点云
            rays_theta, rays_phi = self.livox_generator.sample_ray_angles()
            self.lidar_sim.update_scene(self.model, self.data)
            points = self.lidar_sim.get_lidar_points(rays_phi, rays_theta, self.data)

            lidar_position = self.lidar_sim.sensor_position
            lidar_orientation = Rotation.from_matrix(self.lidar_sim.sensor_rotation).as_quat()
            self.broadcast_tf(self.tf_broadcaster, "world", "body", lidar_position, lidar_orientation)
            self.publish_point_cloud(self.pub_lidar, points, "body")

            # points shape: (3, N) or (N, 3)
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

        time.sleep(0.01)




if __name__ == "__main__":
    cfg = {
        "is_have_moving_base": False,
        "moving_base": {
            "wheel_radius": 0.05,
            "half_wheelbase": 0.2
        },
        "is_have_arm": False,
        "is_use_gs_render": False,
        "episode_len": 100,
        "is_save_record_data": False,
        "camera_names": [],
        "robot_link_list": [],
        "env_name": "MyMobileRobotEnv"
    }
    test = Mid360LidarEnv("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/mid360/scene.xml", cfg)
    test.run_loop()