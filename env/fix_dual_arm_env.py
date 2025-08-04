from env.base_env import *
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import geometry_msgs.msg
import env
from env.se3_controller import *
from env.motor_mixer import *
import threading
from env.utils import np_random
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

class FixDualArmEnv(BaseEnv):
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



        self.last_lidar_pub_time = rospy.Time.now()
        self.lidar_pub_interval = rospy.Duration(0.1)  # 10Hz = 0.1s

      
        
        # self.gs_model_dict["apple"]="/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply"
        # self.gs_renderer.load_single_model("apple", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply")
        self.add_gaussian_model(
            "apple", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply"
        )
        self.add_gaussian_model(
            "banana", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/banana/banana_res.ply"
        )
        self.np_random , self.seed_value = np_random()
       

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



    def reset_model(self):
        self.step_number = 0

        noise_low = 0.0
        noise_high = 0.0  

        qpos =  self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )

        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        # 初始中心位置
        banana_base_pos = np.array([1.08, 1.73, 0.758])
        banana_quat_wxyz = np.array([-0.7071, 0, 0, 0.7071])  # 默认朝向，可调整

        # 在 x 和 y 上加噪声，z 不变
        banana_pos = banana_base_pos.copy()
        banana_pos[0] += self.np_random.uniform(-0.1, 0.1)  # x 随机
        banana_pos[1] += self.np_random.uniform(-0.1, 0.1)  # y 随机

        # 拼接 qpos
        banana_qpos = np.concatenate([banana_pos, banana_quat_wxyz])

        # 设置到 qpos 向量中
        banana_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "banana")
        banana_qpos_addr = self.model.jnt_qposadr[banana_joint_id]
        qpos[banana_qpos_addr:banana_qpos_addr + 7] = banana_qpos

        # 初始中心位置
        apple_base_pos = np.array([1.02, 2.27, 0.768])
        apple_quat_wxyz = np.array([1, 0, 0, 0])  # 默认朝向，可调整

        # 在 x 和 y 上加噪声，z 不变
        apple_pos = apple_base_pos.copy()
        apple_pos[0] += self.np_random.uniform(-0.08, 0.08)  # x 随机
        apple_pos[1] += self.np_random.uniform(-0.1, 0.00)  # y 随机

        # 拼接 qpos
        apple_qpos = np.concatenate([apple_pos, apple_quat_wxyz])

        # 设置到 qpos 向量中
        apple_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "apple")
        apple_qpos_addr = self.model.jnt_qposadr[apple_joint_id]
        qpos[apple_qpos_addr:apple_qpos_addr + 7] = apple_qpos
  
        # reset state
        self.set_state(qpos, qvel)
        # self._set_goal_pose()

        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in self.camera_names},
                'qpos': [],
                'actions': []
            }
        }

        self.q_index = 0
        self.q_vec_list = []

        
        observation, _, _, _, _, _ = self._get_observation()
        return observation
    

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

    def get_cur_robot_state(self):
        '''
        return arm_14d data: 
            [left_arm, left_gripper, right_arm, right_gripper]
        '''
        left_arm_pos, left_grip = self.get_cur_arm_joints_gripper_state("left")
        right_arm_pos, right_grip = self.get_cur_arm_joints_gripper_state("right")
        arm_14d = np.concatenate([left_arm_pos, [left_grip] , right_arm_pos, [right_grip]])
        
        return arm_14d


    # get observation 
    def _get_observation(self):
        #robot stats
        arm_14d = self.get_cur_robot_state()

        self.update_gs_scene()
        
        gs_imgs, depth_imgs, free_img, free_depth_img = self.get_img()
        obs = np.concatenate([arm_14d])

        return obs, arm_14d, gs_imgs, depth_imgs, free_img, free_depth_img

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


        now = rospy.Time.now()
        if now - self.last_lidar_pub_time >= self.lidar_pub_interval:
            self.last_lidar_pub_time = now
            # 发布里程计信息
            self.publish_odometry()

        if rospy.get_param("/debug", False):
            print(f"[CTRL] v_l={v_l:.2f}, v_r={v_r:.2f}")

        time.sleep(0.002)




# if __name__ == "__main__":
#     cfg = {
#         "is_have_moving_base": True,
#         "moving_base": {
#             "wheel_radius": 0.05,
#             "half_wheelbase": 0.2
#         },
#         "is_have_arm": True,
#         "is_use_gs_render": True,
#         "episode_len": 100,
#         "is_save_record_data": True,
#         "camera_names": ["3rd", "wrist_cam_left", "wrist_cam_right", "track"],
#         "robot_link_list": ["left_link1", "left_link2", "left_link3", "left_link4", "left_link5", "left_link6", "left_link7", "left_link8", "right_link1", "right_link2", "right_link3", "right_link4", "right_link5", "right_link6", "right_link7", "right_link8"],
#         "env_name": "MyMobileRobotEnv",
#         "obj_list": ["mobile_ai","desk","apple","banana"]
#     }
#     test = FixDualArmEnv("model_asserts/mujoco_asserts/mobile_ai_robot/scene.xml", cfg)
#     test.run_loop()