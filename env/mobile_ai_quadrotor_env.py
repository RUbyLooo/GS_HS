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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 导入激光雷达包装类和扫描模式生成函数
# from mujoco_lidar.scan_gen import (
#     generate_HDL64,          # Velodyne HDL-64E 模式
#     generate_vlp32,          # Velodyne VLP-32C 模式
#     generate_os128,          # Ouster OS-128 模式
#     LivoxGenerator,          # Livox系列雷达
#     generate_grid_scan_pattern  # 自定义网格扫描模式
# )
# from mujoco_lidar.lidar_wrapper import MjLidarWrapper
# from mujoco_lidar.scan_gen import generate_grid_scan_pattern

class MobileAiQuadrotorEnv(BaseEnv):
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


        # self.global_map = self.load_point_cloud("/home/cfy/cfy/cfy/lerobot_nn/mobile_ai_rl/mobile_ai_gs/models/3dgs/scene/global_map.pcd")
        # self.global_map_pub = rospy.Publisher("/map_points", PointCloud2, queue_size=1)

        # ### livox lidar 相关
        # self.livox_generator = LivoxGenerator("mid360")  # 可选: "avia", "mid40", "mid70", "mid360", "tele"
        # self.rays_theta, self.rays_phi = self.livox_generator.sample_ray_angles()
        # print(f"self.rays_theta : {self.rays_theta}, self.rays_phi : {self.rays_phi}")
        # site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        # print("site_id =", site_id)
        # self.lidar_sim = MjLidarWrapper(
        #     self.model, 
        #     self.data, 
        #     site_name="lidar_site",  # 与MJCF中的<site name="...">匹配
        #     args={
        #         "geom_group": [0],  # 根据你想检测的 group 设置
        #         "enable_profiling": False, # 启用性能分析（可选）
        #         "verbose": False           # 显示详细信息（可选）
        #     }
        # )
        # # 创建TF广播者
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # self.pub_lidar = rospy.Publisher('/livox_points', PointCloud2, queue_size=1)

        self.last_lidar_pub_time = rospy.Time.now()
        self.lidar_pub_interval = rospy.Duration(0.1)  # 10Hz = 0.1s

        # ### 全局点云相关
        # if self.global_map is not None and len(self.global_map) > 0:
            # self.global_map_msg = self.xyz_array_to_pointcloud2(self.global_map, frame_id="world")
            


        ### 四旋翼控制器
        # 初始化SE3控制器
        self.ctrl = SE3Controller()
        # 初始化电机动力分配器
        self.mixer = Mixer()
        # 设置参数
        self.ctrl.kx = 0.6
        self.ctrl.kv = 0.4
        self.ctrl.kR = 6.0
        self.ctrl.kw = 1.0

        self.gravity = 9.8066        # 重力加速度 单位m/s^2
        self.mass = 0.033            # 飞行器质量 单位kg
        self.Ct = 3.25e-4            # 电机推力系数 (N/krpm^2)
        self.Cd = 7.9379e-6          # 电机反扭系数 (Nm/krpm^2)

        self.arm_length = 0.065/2.0  # 电机力臂长度 单位m
        self.max_thrust = 0.1573     # 单个电机最大推力 单位N (电机最大转速22krpm)
        self.max_torque = 3.842e-03  # 单个电机最大扭矩 单位Nm (电机最大转速22krpm)
        self.torque_scale = 0.001    # 控制器控制量到实际扭矩(Nm)的缩放系数(因为是无模型控制所以需要此系数)
        self.dt = 0.002

        self.render_thread = threading.Thread(target=self.render_loop)
        self.render_thread.start()

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

    # 根据电机转速计算电机推力
    def calc_motor_force(self, krpm):
        return self.Ct * krpm**2

    # 根据推力计算电机转速
    def calc_motor_speed_by_force(self, force):
        if force > self.max_thrust:
            force = self.max_thrust
        elif force < 0:
            force = 0
        return np.sqrt(force / self.Ct)

    # 根据扭矩计算电机转速 注意返回数值为转速绝对值 根据实际情况决定转速是增加还是减少
    def calc_motor_speed_by_torque(self, torque):
        if torque > self.max_torque:  # 扭矩绝对值限制
            torque = self.max_torque
        return np.sqrt(torque / self.Cd)

    # 根据电机转速计算电机转速
    def calc_motor_speed(self, force):
        if force > 0:
            return self.calc_motor_speed_by_force(force)

    # 根据电机转速计算电机扭矩
    def calc_motor_torque(self, krpm):
        return self.Cd * krpm**2

    # 根据电机转速计算电机归一化输入
    def calc_motor_input(self, krpm):
        if krpm > 22:
            krpm = 22
        elif krpm < 0:
            krpm = 0
        _force = self.calc_motor_force(krpm)
        _input = _force / self.max_thrust
        if _input > 1:
            _input = 1
        elif _input < 0:
            _input = 0
        return _input

    def render_loop(self):
        rate = rospy.Rate(30)  # 设置为 30Hz
        while not rospy.is_shutdown():
            self.show_gs_render()

    def run_before(self):
        pass

    def run_func(self):
        if self.vx is not None:
            self.vx = 100 * self.vx
        if self.wz is not None:
            self.wz = 100 * self.wz
        v_l, v_r = self.controller.ctrl(self.vx, 0.0, self.wz)
        # 应用控制值
        self.data.ctrl[0] = v_l
        self.data.ctrl[1] = v_r

        # 四旋翼控制量计算
        quadrotor_sensor_data = self.data.sensordata
        quadrotor_gyro_x = quadrotor_sensor_data[0]
        quadrotor_gyro_y = quadrotor_sensor_data[1]
        quadrotor_gyro_z = quadrotor_sensor_data[2]
        quadrotor_acc_x = quadrotor_sensor_data[3]
        quadrotor_acc_y = quadrotor_sensor_data[4]
        quadrotor_acc_z = quadrotor_sensor_data[5]
        quadrotor_quat_w = quadrotor_sensor_data[6]
        quadrotor_quat_x = quadrotor_sensor_data[7]
        quadrotor_quat_y = quadrotor_sensor_data[8]
        quadrotor_quat_z = quadrotor_sensor_data[9]
        quadrotor_quat = np.array([quadrotor_quat_x, quadrotor_quat_y, quadrotor_quat_z, quadrotor_quat_w])  # x y z w
        quadrotor_omega = np.array([quadrotor_gyro_x, quadrotor_gyro_y, quadrotor_gyro_z])  # 角速度

        
        quadrotor_goal_pos, quadrotor_goal_heading = np.array([0.5, 0.0, 0.5]), np.array([1.0, 0.0, 0.0])  # 目标位置
        lidar_pos, lidar_quat = self.get_site_pos_ori("lidar_site")
        lidar_pos = np.array(lidar_pos)
        lidar_quat = np.array(lidar_quat)
        quadrotor_goal_pos = lidar_pos
        r = Rotation.from_quat(lidar_quat)
        heading = r.apply([1.0, 0.0, 0.0])  # 以 x 轴作为朝向基准

        quadrotor_goal_heading = heading  # shape: (3,)

        quadrotor_goal_vel = np.array([0, 0, 0])              # 目标速度
        quadrotor_goal_quat = np.array([1.0, 0.0, 0.0, 0.0])     # 目标四元数(无用)
        quadrotor_goal_omega = np.array([0, 0, 0])            # 目标角速度
        quadrotor_goal_state = State(quadrotor_goal_pos, quadrotor_goal_vel, quadrotor_goal_quat, quadrotor_goal_omega)

        # 构建四旋翼当前状态
        quadrotor_pos, quadrotor_vel, _, _ = self.get_body_qpos_qvel("cf2")
        quadrotor_curr_state = State(quadrotor_pos, quadrotor_vel, quadrotor_quat, quadrotor_omega)

        # 更新控制器
        quadrotor_forward = quadrotor_goal_heading
        quadrotor_control_command = self.ctrl.control_update(quadrotor_curr_state, quadrotor_goal_state, self.dt, quadrotor_forward)
        quadrotor_ctrl_thrust = quadrotor_control_command.thrust    # 总推力控制量(mg为单位)
        quadrotor_ctrl_torque = quadrotor_control_command.angular   # 三轴扭矩控制量

        # Mixer
        quadrotor_mixer_thrust = quadrotor_ctrl_thrust * self.gravity * self.mass     # 机体总推力(N)
        quadrotor_mixer_torque = quadrotor_ctrl_torque * self.torque_scale       # 机体扭矩(Nm)
        # 输出到电机
        quadrotor_motor_speed = self.mixer.calculate(quadrotor_mixer_thrust, quadrotor_mixer_torque[0], quadrotor_mixer_torque[1], quadrotor_mixer_torque[2]) # 动力分配
        self.data.actuator('motor1').ctrl[0] = self.calc_motor_input(quadrotor_motor_speed[0])
        self.data.actuator('motor2').ctrl[0] = self.calc_motor_input(quadrotor_motor_speed[1])
        self.data.actuator('motor3').ctrl[0] = self.calc_motor_input(quadrotor_motor_speed[2])
        self.data.actuator('motor4').ctrl[0] = self.calc_motor_input(quadrotor_motor_speed[3])


        ### 电梯控制


        now = rospy.Time.now()
        if now - self.last_lidar_pub_time >= self.lidar_pub_interval:
            self.last_lidar_pub_time = now
            # 发布里程计信息
            self.publish_odometry()
            
        #     rays_theta, rays_phi = self.livox_generator.sample_ray_angles()
        #     # 更新mujoco中激光雷达的场景
        #     self.lidar_sim.update_scene(self.model, self.data)
        #     # 获取激光雷达点云
        #     points = self.lidar_sim.get_lidar_points(rays_phi, rays_theta, self.data)
        #     # 获取激光雷达位置和方向
        #     lidar_position = self.lidar_sim.sensor_position
        #     lidar_orientation = Rotation.from_matrix(self.lidar_sim.sensor_rotation).as_quat()
        #     # 广播激光雷达的TF
        #     self.broadcast_tf(self.tf_broadcaster, "world", "body", lidar_position, lidar_orientation)
        #     self.publish_point_cloud(self.pub_lidar, points, "body")

        # self.show_gs_render()

        if rospy.get_param("/debug", False):
            print(f"[CTRL] v_l={v_l:.2f}, v_r={v_r:.2f}")

        time.sleep(0.002)




if __name__ == "__main__":
    cfg = {
        "is_have_moving_base": True,
        "moving_base": {
            "wheel_radius": 0.05,
            "half_wheelbase": 0.2
        },
        "is_have_arm": True,
        "is_use_gs_render": True,
        "episode_len": 100,
        "is_save_record_data": True,
        "camera_names": ["3rd", "wrist_cam_left", "wrist_cam_right", "track"],
        "robot_link_list": ["left_link1", "left_link2", "left_link3", "left_link4", "left_link5", "left_link6", "left_link7", "left_link8", "right_link1", "right_link2", "right_link3", "right_link4", "right_link5", "right_link6", "right_link7", "right_link8"],
        "env_name": "MyMobileRobotEnv",
        "obj_list":["mobile_ai","cf2"]
    }
    test = MobileAiQuadrotorEnv("model_asserts/mujoco_asserts/mobile_ai_quadrotor_robot/scene.xml", cfg)
    test.run_loop()