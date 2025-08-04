import numpy as np
import torch
import time
import rerun as rr
# 导入机器人模型
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
    moving_dual_piper,
    sim_single_piper,
)
# 导入 util
from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from dataclasses import asdict, dataclass
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
# 导入 policy
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# from lerobot.common.policies.actvla.modeling_actvla import ACTvlaPolicy
# from lerobot.common.policies.vggtvla.modeling_vggtvla import VGGTVLAPolicy
# from lerobot.common.policies.qwenvla.modeling_qwenvla import QwenVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.configs import parser
# piper sdk
from piper_sdk import *
import cv2

import threading
action_lock = threading.Lock()
last_action = None  # 会存 tensor，初始化后由推理线程第一次写入

@dataclass
class InferenceConfig:
    robot: RobotConfig
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    ckpt_path: str = None
    task: str = None

    def __post_init__(self):

        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def enable_fun(piper:C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)

def sim_loop(robot, physical_fps: int):
    
    dt = 1.0 / physical_fps
    global last_action

    # 确保有一个初始动作
    if last_action is None:
        last_action = np.zeros(7, dtype=np.float32)

    while True:
        # 安全地复制最新动作
        with action_lock:
            a = last_action.copy()   # 用 numpy.copy()

        # 一直在跑物理引擎
        robot.sim_action(a)
        time.sleep(dt)


def infer_loop(cfg, robot, policy, inference_interval: float):
    global last_action
    first = True

   
       
    while True:
        if first:
            time.sleep(1.0)
            first = False
        # 拿最新观测
        obs = robot.get_observation()
        observation_frame = {}

        # 用于保存 state 数值
        state_values = []

        for key, value in obs.items():
            # 把所有带 .pos 的 float/ndarray 合并成 state
            if key.endswith('.pos'):
                state_values.append(np.float32(value))

            # 把值为 HWC 格式的 ndarray（图像）保存为 observation.images.{key}
            elif isinstance(value, np.ndarray) and value.ndim == 3:
                observation_frame[f'observation.images.{key}'] = value

        # 添加合并后的状态
        observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)
       
        for name in observation_frame:
            if "image" in name:
                # 图像预处理
                obs = observation_frame[name].astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
            else:
                obs = observation_frame[name]  # 例如 state 是 np.ndarray

            # 增加 batch 维度
            obs = np.expand_dims(obs, axis=0)
            obs = np.expand_dims(obs, axis=0)
            observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device="cuda")
        observation_frame["task"] = [cfg.task]


        # 推理，得到 torch.Tensor
        with torch.inference_mode():
            new_a = policy.select_action(observation_frame)  # Tensor of shape [1, action_dim] 或 [action_dim]

        # 转成 numpy 并更新全局
        new_a_np = new_a.cpu().numpy().squeeze().astype(np.float32)
        with action_lock:
            last_action = new_a_np

        time.sleep(inference_interval)

@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    # 创建 robot
    robot = make_robot_from_config(cfg.robot)
    if not robot.is_connected:
        robot.connect()

    if cfg.robot.type == "moving_dual_piper":
        import rospy
        from geometry_msgs.msg import Twist
        rospy.init_node("pub_cmd_vel", anonymous=True)
        cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        print(f"using moving_dual_piper robot, ROS node 'pub_cmd_vel' initialized.")

    inference_time_s = 10000
    fps = 30
    # device choice
    device = "cuda"
    # 创建 policy
    ckpt_path = cfg.ckpt_path
    print(f"ckpt_path : {ckpt_path}")
    if cfg.task == None:
        raise ValueError("You need to provide a task name.")

    # 创建 policy
    if cfg.policy.type == "act":
        policy = ACTPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(ckpt_path)
    elif cfg.policy.type == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(ckpt_path)
    # elif cfg.policy.type == "actvla":
    #     policy = ACTvlaPolicy.from_pretrained(ckpt_path)
    # elif cfg.policy.type == "vggtvla":
    #     policy = VGGTVLAPolicy.from_pretrained(ckpt_path)
    # elif cfg.policy.type == "qwenvla":
    #     policy = QwenVLAPolicy.from_pretrained(ckpt_path)
    else:
        raise ValueError("You need to provide a valid policy between act/diffusion/smolvla.")
    
    # 将 policy 设置为推理模式
    policy.eval()
    policy.to(device)
    policy.reset()
    print(f"[Info] task: {cfg.task}")
    # 缩放因子
    factor = 57324.840764 # 1000 * 180 / 3.14
    if cfg.robot.type == "single_piper":
        # 手臂使能 
        robot.piper.EnableArm(7)
        enable_fun(piper=robot.piper)
        robot.piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
        # 等待使能完成
        time.sleep(2.0)
    elif cfg.robot.type == "dual_piper" or cfg.robot.type == "moving_dual_piper":
        # 左手臂使能
        robot.piper_left.EnableArm(7)
        enable_fun(piper=robot.piper_left)
        robot.piper_left.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        robot.piper_left.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
        # 右手臂使能
        robot.piper_right.EnableArm(7)
        enable_fun(piper=robot.piper_right)
        robot.piper_right.MotionCtrl_2(0x01, 0x01, 60, 0x00)
        robot.piper_right.GripperCtrl(round(1.0 * 70 * 1000), 1000, 0x01, 0)
        # 等待使能完成
        time.sleep(2.0)
    elif cfg.robot.type =="sim_single_piper":
        print("Sim piper using")
    else:
        raise ValueError("Enable arm failed ! You need to provide a valid robot type between single_piper/dual_piper/moving_dual_piper.")

    
    
    # 重置机器人环境
    robot.env.reset()
    # 开启物理仿真引擎
    robot.env.resume_physical_thread()
    rr.init("inference", spawn=True)
    time.sleep(1.0)  # 等待物理线程启动




    # 开始执行推理
    for _ in range(inference_time_s * fps):
        
        start_loop_t = time.perf_counter()
        observation = robot.get_observation()

        # print(f"!!!!!!!!!! observation : {observation}")
        observation_frame = {}

        # 用于保存 state 数值
        state_values = []

        for key, value in observation.items():
            # 把所有带 .pos 的 float/ndarray 合并成 state
            if key.endswith('.pos'):
                state_values.append(np.float32(value))

            # 把值为 HWC 格式的 ndarray（图像）保存为 observation.images.{key}
            elif isinstance(value, np.ndarray) and value.ndim == 3:
                observation_frame[f'observation.images.{key}'] = value

        # 添加合并后的状态
        observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)

        for name in observation_frame:
            if "image" in name:
                # 图像预处理
                obs = observation_frame[name].astype(np.float32) / 255.0
                obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
            else:
                obs = observation_frame[name]  # 例如 state 是 np.ndarray

            # 增加 batch 维度
            obs = np.expand_dims(obs, axis=0)
            observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=device)

        for obs_, val_ in observation.items():
            if isinstance(val_, float):
                rr.log(f"observation.{obs_}", rr.Scalar(val_))
                print(f"{obs_}: {val_} (type: float)")

            elif isinstance(val_, np.ndarray):
                rr.log(f"observation.{obs_}", rr.Image(val_), static=True)
                print(f"{obs_} shape: {val_.shape}, dtype: {val_.dtype}")

            elif isinstance(val_, torch.Tensor):
                rr.log(f"observation.{obs_}", rr.Tensor(val_))
                print(f"{obs_} shape: {val_.shape}, dtype: {val_.dtype}")

            else:
                print(f"{obs_}: unsupported type {type(val_)}")

        infer_start_time = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(observation_frame)
            action = action.squeeze(0).cpu().numpy()            # 去掉 batch 维，转到 cpu，再转 numpy
            print(f"action : {action}")
            robot.env.set_desired_action(action)



        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
           
            

        # numpy_action = action.squeeze(0).cpu().numpy()  # 去掉 batch 维，转到 cpu，再转 numpy
        # position = numpy_action.tolist()  # 转成 python list，方便后续使用


        # # 各关节运动限位
        # joint_limits = [(-3, 3)] * 6
        # joint_limits[0] = (-2.687, 2.687)
        # joint_limits[1] = (0.0, 3.403) 
        # joint_limits[2] = (-3.0541012, 0.0) 
        # joint_limits[3] = (-1.5499, 1.5499) 
        # joint_limits[4] = (-1.22, 1.22) 
        # joint_limits[5] = (-1.7452, 1.7452) 
        # # Clamp position values to joint limits
        # def clamp(value, min_val, max_val):
        #     return max(min(value, max_val), min_val)

        # if cfg.robot.type == "single_piper":
        #     joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
        #     joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
        #     joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
        #     joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
        #     joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
        #     joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
        #     joint_6 = round(position[6] * 70 * 1000)
        #     robot.piper.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
        #     robot.piper.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
        # elif cfg.robot.type == "dual_piper":
        #     # Left arm control (position 0-5)
        #     left_joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
        #     left_joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
        #     left_joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
        #     left_joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
        #     left_joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
        #     left_joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
        #     left_joint_6 = round(position[6] * 70 * 1000)
        #     robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
        #     robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)

        #     # Right arm control (position 7-12)
        #     right_joint_0 = round(clamp(position[7], joint_limits[0][0], joint_limits[0][1]) * factor)
        #     right_joint_1 = round(clamp(position[8], joint_limits[1][0], joint_limits[1][1]) * factor)
        #     right_joint_2 = round(clamp(position[9], joint_limits[2][0], joint_limits[2][1]) * factor)
        #     right_joint_3 = round(clamp(position[10], joint_limits[3][0], joint_limits[3][1]) * factor)
        #     right_joint_4 = round(clamp(position[11], joint_limits[4][0], joint_limits[4][1]) * factor)
        #     right_joint_5 = round(clamp(position[12], joint_limits[5][0], joint_limits[5][1]) * factor)
        #     right_joint_6 = round(position[13] * 70 * 1000)
        #     robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
        #     robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        # elif cfg.robot.type == "moving_dual_piper" and not rospy.is_shutdown():
        #     # scout_mini control (position 0-1)
        #     twist_msg = Twist()
        #     twist_msg.linear.x = position[0]   # 线速度
        #     twist_msg.angular.z = position[1]  # 角速度
        #     cmd_vel_pub.publish(twist_msg)
        #     # Left arm control (position 2-8)
        #     left_joint_0 = round(clamp(position[2], joint_limits[0][0], joint_limits[0][1]) * factor)
        #     left_joint_1 = round(clamp(position[3], joint_limits[1][0], joint_limits[1][1]) * factor)
        #     left_joint_2 = round(clamp(position[4], joint_limits[2][0], joint_limits[2][1]) * factor)
        #     left_joint_3 = round(clamp(position[5], joint_limits[3][0], joint_limits[3][1]) * factor)
        #     left_joint_4 = round(clamp(position[6], joint_limits[4][0], joint_limits[4][1]) * factor)
        #     left_joint_5 = round(clamp(position[7], joint_limits[5][0], joint_limits[5][1]) * factor)
        #     left_joint_6 = round(position[8] * 70 * 1000)
        #     robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
        #     robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)

        #     # Right arm control (position 9-15)
        #     right_joint_0 = round(clamp(position[9], joint_limits[0][0], joint_limits[0][1]) * factor)
        #     right_joint_1 = round(clamp(position[10], joint_limits[1][0], joint_limits[1][1]) * factor)
        #     right_joint_2 = round(clamp(position[11], joint_limits[2][0], joint_limits[2][1]) * factor)
        #     right_joint_3 = round(clamp(position[12], joint_limits[3][0], joint_limits[3][1]) * factor)
        #     right_joint_4 = round(clamp(position[13], joint_limits[4][0], joint_limits[4][1]) * factor)
        #     right_joint_5 = round(clamp(position[14], joint_limits[5][0], joint_limits[5][1]) * factor)
        #     right_joint_6 = round(position[15] * 70 * 1000)
        #     robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
        #     robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
        # else:
        #     raise ValueError("Execute action failed ! You need to provide a valid robot type between single_piper/dual_piper/moving_dual_piper.")

        
        # busy_wait(0.002)












    # TODO -------------------------------------------------------------------------
    # max_wait_time = 5.0  # 最多等待 5 秒
    # start_time = time.perf_counter()
    # # while True:
    # #     if hasattr(robot, "env") and hasattr(robot.env, "gs_renderer") and robot.env.gs_renderer is not None:
    # #         try:
    # #             # 尝试渲染一帧图像，确保 renderer 已准备好
    # #             _ = robot.env.gs_renderer.render()
    # #             print("GS renderer is ready.")
    # #             break
    # #         except Exception as e:
    # #             print(f"GS renderer not ready yet: {e}")
        
    # #     if time.perf_counter() - start_time > max_wait_time:
    # #         raise TimeoutError("GS renderer initialization timed out.")
    
    # # time.sleep(0.1)  # 小延迟防止死循环卡 CPU

    # # 先执行第一帧动作， 获取第一帧观测
    # action = np.zeros(7)
    # robot.env.step(action)
    # fir_obs = robot.get_observation()

    # if fir_obs is None:
    #     raise TimeoutError(" 第一帧动作初始化失败. ")
    






    # ## 增加一个，使得物理引擎一直在跑

    # for _ in range(inference_time_s * fps):
    #     print(f"cout : {cout}")
    #     cout = cout + 1
    #     start_loop_t = time.perf_counter()
    #     # # 得到当前观测 : (state/images)

    #     wait_start = time.perf_counter()
    #     # while robot.env.latest_images is None:
    #     #     if time.perf_counter() - wait_start > 3.0:
    #     #         raise TimeoutError("Waited too long for GS renderer to produce first image.")
    #     #     time.sleep(0.01)

    #     observation = robot.get_observation()
    #     # print(f"observation is : {observation}")
    #     observation_frame = {}

    #     # 用于保存 state 数值
    #     state_values = []

    #     for key, value in observation.items():
    #         # 把所有带 .pos 的 float/ndarray 合并成 state
    #         if key.endswith('.pos'):
    #             state_values.append(np.float32(value))

    #         # 把值为 HWC 格式的 ndarray（图像）保存为 observation.images.{key}
    #         elif isinstance(value, np.ndarray) and value.ndim == 3:
    #             observation_frame[f'observation.images.{key}'] = value

    #     # 添加合并后的状态
    #     observation_frame['observation.state'] = np.array(state_values, dtype=np.float32)

    #     for name in observation_frame:
    #         if "image" in name:
    #             # 图像预处理
    #             obs = observation_frame[name].astype(np.float32) / 255.0
    #             obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
    #         else:
    #             obs = observation_frame[name]  # 例如 state 是 np.ndarray

    #         # 增加 batch 维度
    #         obs = np.expand_dims(obs, axis=0)
    #         observation_frame[name] = torch.tensor(obs, dtype=torch.float32, device=device)
    #     # if cfg.task == None:
    #     #     raise ValueError("You need to provide a task name.")
    #     # observation_frame["task"] = [cfg.task]
    #     # rr.log("observation.3rd", rr.Image(observation["3rd"]), static=True)  # 明确写 key 和 value
    #     # rr.log("observation.wrist", rr.Image(observation["wrist"]), static=True)
    #     cv2.imwrite("images/debug_3rd.png", observation["3rd"])
    #     cv2.imwrite("images/debug_wrist.png", observation["wrist"])
    
    #     for obs_, val_ in observation.items():
    #         if isinstance(val_, float):
    #             rr.log(f"observation.{obs_}", rr.Scalar(val_))
    #             print(f"{obs_} shape: {val_.shape}, dtype: {val_.dtype}")

    #         elif isinstance(val_, np.ndarray):
    #             rr.log(f"observation.{obs_}", rr.Image(val_), static=True)
    #             print(f"{obs_} shape: {val_.shape}, dtype: {val_.dtype}")

    #     for i in range(len(state_values)):
          
    #         rr.log(f"action.{i}", rr.Scalar(state_values[i]))

        

    #     infer_start_time = time.perf_counter()
    #     with torch.inference_mode():
    #         action = policy.select_action(observation_frame)
    #         # action = [np.array([1,0,1,0,1,0,1])]
    #         # print(f"inference time is :{time.perf_counter()-infer_start_time}")
    #         robot.sim_action(action)
           
            

    #     # numpy_action = action.squeeze(0).cpu().numpy()  # 去掉 batch 维，转到 cpu，再转 numpy
    #     # position = numpy_action.tolist()  # 转成 python list，方便后续使用


    #     # # 各关节运动限位
    #     # joint_limits = [(-3, 3)] * 6
    #     # joint_limits[0] = (-2.687, 2.687)
    #     # joint_limits[1] = (0.0, 3.403) 
    #     # joint_limits[2] = (-3.0541012, 0.0) 
    #     # joint_limits[3] = (-1.5499, 1.5499) 
    #     # joint_limits[4] = (-1.22, 1.22) 
    #     # joint_limits[5] = (-1.7452, 1.7452) 
    #     # # Clamp position values to joint limits
    #     # def clamp(value, min_val, max_val):
    #     #     return max(min(value, max_val), min_val)

    #     # if cfg.robot.type == "single_piper":
    #     #     joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
    #     #     joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
    #     #     joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
    #     #     joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
    #     #     joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
    #     #     joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
    #     #     joint_6 = round(position[6] * 70 * 1000)
    #     #     robot.piper.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
    #     #     robot.piper.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)
    #     # elif cfg.robot.type == "dual_piper":
    #     #     # Left arm control (position 0-5)
    #     #     left_joint_0 = round(clamp(position[0], joint_limits[0][0], joint_limits[0][1]) * factor)
    #     #     left_joint_1 = round(clamp(position[1], joint_limits[1][0], joint_limits[1][1]) * factor)
    #     #     left_joint_2 = round(clamp(position[2], joint_limits[2][0], joint_limits[2][1]) * factor)
    #     #     left_joint_3 = round(clamp(position[3], joint_limits[3][0], joint_limits[3][1]) * factor)
    #     #     left_joint_4 = round(clamp(position[4], joint_limits[4][0], joint_limits[4][1]) * factor)
    #     #     left_joint_5 = round(clamp(position[5], joint_limits[5][0], joint_limits[5][1]) * factor)
    #     #     left_joint_6 = round(position[6] * 70 * 1000)
    #     #     robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
    #     #     robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)

    #     #     # Right arm control (position 7-12)
    #     #     right_joint_0 = round(clamp(position[7], joint_limits[0][0], joint_limits[0][1]) * factor)
    #     #     right_joint_1 = round(clamp(position[8], joint_limits[1][0], joint_limits[1][1]) * factor)
    #     #     right_joint_2 = round(clamp(position[9], joint_limits[2][0], joint_limits[2][1]) * factor)
    #     #     right_joint_3 = round(clamp(position[10], joint_limits[3][0], joint_limits[3][1]) * factor)
    #     #     right_joint_4 = round(clamp(position[11], joint_limits[4][0], joint_limits[4][1]) * factor)
    #     #     right_joint_5 = round(clamp(position[12], joint_limits[5][0], joint_limits[5][1]) * factor)
    #     #     right_joint_6 = round(position[13] * 70 * 1000)
    #     #     robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
    #     #     robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
    #     # elif cfg.robot.type == "moving_dual_piper" and not rospy.is_shutdown():
    #     #     # scout_mini control (position 0-1)
    #     #     twist_msg = Twist()
    #     #     twist_msg.linear.x = position[0]   # 线速度
    #     #     twist_msg.angular.z = position[1]  # 角速度
    #     #     cmd_vel_pub.publish(twist_msg)
    #     #     # Left arm control (position 2-8)
    #     #     left_joint_0 = round(clamp(position[2], joint_limits[0][0], joint_limits[0][1]) * factor)
    #     #     left_joint_1 = round(clamp(position[3], joint_limits[1][0], joint_limits[1][1]) * factor)
    #     #     left_joint_2 = round(clamp(position[4], joint_limits[2][0], joint_limits[2][1]) * factor)
    #     #     left_joint_3 = round(clamp(position[5], joint_limits[3][0], joint_limits[3][1]) * factor)
    #     #     left_joint_4 = round(clamp(position[6], joint_limits[4][0], joint_limits[4][1]) * factor)
    #     #     left_joint_5 = round(clamp(position[7], joint_limits[5][0], joint_limits[5][1]) * factor)
    #     #     left_joint_6 = round(position[8] * 70 * 1000)
    #     #     robot.piper_left.JointCtrl(left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5)
    #     #     robot.piper_left.GripperCtrl(abs(left_joint_6), 1000, 0x01, 0)

    #     #     # Right arm control (position 9-15)
    #     #     right_joint_0 = round(clamp(position[9], joint_limits[0][0], joint_limits[0][1]) * factor)
    #     #     right_joint_1 = round(clamp(position[10], joint_limits[1][0], joint_limits[1][1]) * factor)
    #     #     right_joint_2 = round(clamp(position[11], joint_limits[2][0], joint_limits[2][1]) * factor)
    #     #     right_joint_3 = round(clamp(position[12], joint_limits[3][0], joint_limits[3][1]) * factor)
    #     #     right_joint_4 = round(clamp(position[13], joint_limits[4][0], joint_limits[4][1]) * factor)
    #     #     right_joint_5 = round(clamp(position[14], joint_limits[5][0], joint_limits[5][1]) * factor)
    #     #     right_joint_6 = round(position[15] * 70 * 1000)
    #     #     robot.piper_right.JointCtrl(right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5)
    #     #     robot.piper_right.GripperCtrl(abs(right_joint_6), 1000, 0x01, 0)
    #     # else:
    #     #     raise ValueError("Execute action failed ! You need to provide a valid robot type between single_piper/dual_piper/moving_dual_piper.")

    #     # dt_s = time.perf_counter() - start_loop_t
    #     # busy_wait(1 / fps - dt_s)
    #     # busy_wait(0.002)
    #     time.sleep(0.002)


if __name__ == "__main__":
    inference()