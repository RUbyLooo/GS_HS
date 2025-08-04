from env.base_sim_env import BaseMujocoGSWorker
import gym
from gym import spaces
import time
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from gym.vector import AsyncVectorEnv
from easydict import EasyDict
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import torch
import torch.nn as nn


## pre

class SinglePiperMujocoGSWorker(BaseMujocoGSWorker):
    def __init__(self, config):
        super().__init__(config)


    def _get_simulation_data(self) -> dict:
        """
        返回当前仿真状态，包含 robot、item、camera 的完整数据
        """
        sim_data = {}

        if self.running == False:
            return sim_data

        # 获取当前机器人的观测
        sim_data["robot"] = []
        for robot in self.config.robot:
            robot_name = robot["name"]
            joint_list = robot["joint_list"]
            body_list = robot["body_list"]

            robot_joints = []

            for joint_name, body_name in zip(joint_list, body_list):
                # 获取关节传感器数据
                joint_pos = self._get_sensor_data(f"{joint_name}_pos")[0]  # 关节角度
                joint_vel = self._get_sensor_data(f"{joint_name}_vel")[0]  # 关节角速度

                # 获取 body 位姿
                body_pos, body_quat = self._get_body_pose(body_name)

                # 填充每个关节信息
                joint_info = {
                    "name": body_name,
                    "joint_pos": joint_pos,
                    "joint_vel": joint_vel,
                    "pos": body_pos.tolist(),
                    "quat_wxyz": body_quat.tolist()
                }
                robot_joints.append(joint_info)

            sim_data["robot"].append({
                "name": robot_name,
                "joint_list": robot_joints
            })
        
        # 获取当前交互物品的观测
        sim_data["item"] = []
        for item in self.config.item:
            item_name = item["name"]

            # 获取物品的 body 位姿
            body_pos, body_quat = self._get_body_pose(item_name)

            item_info = {
                "name": item_name,
                "pos": body_pos.tolist(),          # 转成 list 存储
                "quat_wxyz": body_quat.tolist()    # 转成 list 存储
            }
            sim_data["item"].append(item_info)

        # 获取当前相机的位姿
        camera_list = []
        for cam_id in range(self.model.ncam):
            cam_name = self.model.camera(cam_id).name
            cam_pos = self.data.cam_xpos[cam_id].tolist()  # 转成 list，和你取的时候格式一致
            cam_rot = self.data.cam_xmat[cam_id].reshape((3, 3))
            cam_quat = Rotation.from_matrix(cam_rot).as_quat().tolist()  # 转成 list

            camera_list.append({
                "name": cam_name,
                "fovy": self.model.cam_fovy[cam_id],  # 获取相机的视场角
                "pos": cam_pos,
                "quat": cam_quat
            })
        sim_data["camera"] = camera_list

        return sim_data
        
class MujocoGSGymEnv(gym.Env):
    def __init__(self, worker_config):
        super(MujocoGSGymEnv, self).__init__()
        # 启动一个独立的 worker
        self.worker = SinglePiperMujocoGSWorker(worker_config)
        # 打印环境信息
        self.worker._print_all_joint_info()
        self.worker._print_all_body_info()

        self._reset_noise_scale = 1e-2
        self.init_qpos = np.zeros(6)
        self.init_qvel = np.zeros(6)
        self.episode_len = 200

        # 各关节运动限位
        self.joint_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
        ])
        
        # 定义 action 和 observation 空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(worker_config.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "joint_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(worker_config.joint_dim,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(3, worker_config.gs_config.gs_rgb_height, worker_config.gs_config.gs_rgb_width), dtype=np.uint8)       
        })

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=6
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=6
        )
        
        self._set_state(qpos, qvel)
        self.worker.is_reset = True
        obs = self._get_observation()

        obs_res = self._process_obs(obs)
        self.worker.is_reset = False
        return obs_res, {}

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def _set_state(self, qpos, qvel):
        # assert qpos.shape == (6,) and qvel.shape == (6,)
        self.worker.data.qpos[:6] = np.copy(qpos)
        self.worker.data.qvel[:6] = np.copy(qvel)
        # mujoco 仿真向前推进一步
        mujoco.mj_forward(self.worker.model, self.worker.data)
        # 更新一下 mujoco
        self.worker._phy_simulation_step()

    def step(self, action):
        # 应用动作
        self._apply_action(action)
        # 提取观测
        obs = self._get_observation()
        obs_res = self._process_obs(obs)

        # 计算 reward、done，示例：
        reward = 0.0
        done = False
        info = {}
        truncated = {}

        return obs_res, reward, done, truncated, info
    
    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到每个关节的具体角度范围。

        Args:
            action (np.ndarray): 形状为 (6,) 的数组，值范围在 [-1, 1]

        Returns:
            np.ndarray: 形状为 (6,) 的数组，映射到实际关节角度范围，类型为 numpy.ndarray
        """

        normalized = (action + 1) / 2
        lower_bounds = self.joint_limits[:, 0]
        upper_bounds = self.joint_limits[:, 1]
        # 插值计算
        mapped_action = lower_bounds + normalized * (upper_bounds - lower_bounds)

        return mapped_action

    def _apply_action(self, action):
        mapped_action = self.map_action_to_joint_limits(action)
        self.worker.data.ctrl[:6] = mapped_action
        self.worker._phy_simulation_step()

    def _get_observation(self):
        obs = self.worker._get_cur_obs()
        return obs
    
    def _process_obs(self, raw_obs):
        # 取 joint_pos
        joint_list = raw_obs['robot'][0]['joint_list']
        joint_pos = np.array([joint['joint_pos'] for joint in joint_list[:6]], dtype=np.float32)

        # 取 wrist_cam 的 rgb，并转成 (3, H, W)
        rgb_image = raw_obs['gs_img']['wrist']['rgb']  # 原始是 (H, W, 3)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))  # 转成 (3, H, W)

        return {
            'joint_pos': joint_pos,
            'image': rgb_image
        }

    def close(self):
        self.worker.stop()



if __name__ == "__main__":
    cfg = EasyDict({
        "mujoco_config": {
            "model_path": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/scene.xml", 
        },
        "display_data": True,
        "enable_free_camera": True, 
        "action_dim": 6,
        "joint_dim": 6,
        "gs_config": {
            "gs_rgb_width": 640,  # 你需要设置图像宽度
            "gs_rgb_height": 480,  # 你需要设置图像高度
            "gs_model_dict": {
                "background": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/scene/1lou_0527_res.ply",
                "link1": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link1_rot.ply",
                "link2": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link2_rot.ply",
                "link3": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link3_rot.ply",
                "link4": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link4_rot.ply",
                "link5": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link5_rot.ply",
                "link6": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link6_rot.ply",
                "link7": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link7_rot.ply",
                "link8": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link8_rot.ply",
                "desk": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/desk/1louzhuozi_res.ply",

                "apple": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply",
                "banana": "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/banana/banana_res.ply",
            }
        }, 

        "robot": [
            {
                "name": "piper",  # 机器人名称
                "joint_list": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"],
                "body_list": ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]
            }
        ],
        "item": [
            {"name": "apple"},
            {"name": "banana"},
            {"name": "desk"},
        ],
    })

    
    env = make_vec_env(lambda: MujocoGSGymEnv(cfg), n_envs=1)


    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    """
    训练参数
    参数名	                   解释	                                建议与注意事项
    learning_rate	          学习率	                           固定值适合起步，调 schedule 可提升稳定性
    n_steps	                  每个环境每次 rollout 的步数            必须满足 n_steps * n_envs > 1, 推荐设为 128~2048
    batch_size	              每次优化的最小 batch 大小	             
    n_epochs	              每次更新重复训练的次数	              增加样本利用率， 3~10 是常用区间
    gamma	                  奖励折扣因子（长期 vs 短期）	          0.95~0.99 之间，任务长期性越强设得越高
    device	                  训练使用设备	                        GPU / CPU
    tensorboard_log           训练日志保存地址
    """
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=10,
        batch_size=500,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./ppo_piper/"
    )

    """
    参数名	                  解释	                                   
    total_timesteps          总共与环境交互的步数（env.step() 的次数总和）   
    progress_bar             是否显示训练进度条
    """
    model.learn(total_timesteps=2000*10000, progress_bar=True)
    model.save("piper_ik_ppo_model")

    print(" model sava success ! ")


    # env = MujocoGSGymEnv(cfg)
    # while True:
    #     # 获取动作维度
    #     action_dim = env.worker.model.nu

    #     # 获取每个 actuator 的控制范围
    #     ctrl_range = env.worker.model.actuator_ctrlrange  # shape: (action_dim, 2)

    #     # 按控制范围随机采样动作
    #     action = np.random.uniform(low=ctrl_range[:, 0], high=ctrl_range[:, 1])
    #     env.step(action)
