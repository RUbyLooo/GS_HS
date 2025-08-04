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

        # 默认速度
        self.vx = 0.0
        self.wz = 0.0

        # 控制器初始化
        self.controller = DifferentialDriveWheelController(
            cfg["moving_base"]["wheel_radius"],
            cfg["moving_base"]["half_wheelbase"]
        )

        # ROS 节点和订阅器初始化
        rospy.init_node("mobile_ai_robot_env", anonymous=True)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)
        print("[ROS] Subscribed to /cmd_vel")
        self.odom_pub = rospy.Publisher("/localization", Odometry, queue_size=10)


        self.global_map = self.load_point_cloud("/home/cfy/cfy/cfy/lerobot_nn/mobile_ai_rl/mobile_ai_gs/models/3dgs/scene/global_map.pcd")
        self.global_map_pub = rospy.Publisher("/map_points", PointCloud2, queue_size=1)

        ### lidar 相关
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

    def cmd_vel_callback(self, msg):
        self.vx = msg.linear.x
        self.wz = msg.angular.z
        if rospy.get_param("/debug", False):
            print(f"[CMD_VEL] Received: vx={self.vx}, wz={self.wz}")

    def publish_odometry(self):
        pos, quat = self.get_localization()  # [x,y,z], [w,x,y,z]
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "body"

        odom.pose.pose.position.x = pos[0]
        odom.pose.pose.position.y = pos[1]
        odom.pose.pose.position.z = pos[2]

        odom.pose.pose.orientation.w = quat[0]
        odom.pose.pose.orientation.x = quat[1]
        odom.pose.pose.orientation.y = quat[2]
        odom.pose.pose.orientation.z = quat[3]

        self.odom_pub.publish(odom)

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
        # 通过控制器转换为左右轮速度
        # self.vx = 0.0
        # self.wz = 0.0
        # self.wz = 2.0
        v_l, v_r = self.controller.ctrl(self.vx, 0.0, self.wz)
        # 应用控制值
        self.data.ctrl[0] = v_l
        self.data.ctrl[1] = v_r

        now = rospy.Time.now()
        if now - self.last_lidar_pub_time >= self.lidar_pub_interval:
            self.last_lidar_pub_time = now
            # 发布里程计信息
            self.publish_odometry()
            
            rays_theta, rays_phi = self.livox_generator.sample_ray_angles()
            # 更新mujoco中激光雷达的场景
            self.lidar_sim.update_scene(self.model, self.data)
            # 获取激光雷达点云
            points = self.lidar_sim.get_lidar_points(rays_phi, rays_theta, self.data)
            # 获取激光雷达位置和方向
            lidar_position = self.lidar_sim.sensor_position
            lidar_orientation = Rotation.from_matrix(self.lidar_sim.sensor_rotation).as_quat()
            # 广播激光雷达的TF
            self.broadcast_tf(self.tf_broadcaster, "world", "body", lidar_position, lidar_orientation)
            self.publish_point_cloud(self.pub_lidar, points, "body")
        # 发布全局点云
        # if self.global_map is not None and len(self.global_map) > 0:
        #     msg = self.xyz_array_to_pointcloud2(self.global_map, frame_id="world")
        #     self.global_map_pub.publish(msg)

        if rospy.get_param("/debug", False):
            print(f"[CTRL] v_l={v_l:.2f}, v_r={v_r:.2f}")

        time.sleep(0.0002)




if __name__ == "__main__":
    cfg = {
        "is_have_moving_base": True,
        "moving_base": {
            "wheel_radius": 0.05,
            "half_wheelbase": 0.2
        },
        "is_have_arm": False,
        "is_use_gs_render": False,
        "episode_len": 100,
        "is_save_record_data": True,
        "camera_names": ["cam0", "cam1"],
        "robot_link_list": [],
        "env_name": "MyMobileRobotEnv"
    }
    test = MobileAiRobotEnv("/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/mobile_ai_robot/scene.xml", cfg)
    test.run_loop()