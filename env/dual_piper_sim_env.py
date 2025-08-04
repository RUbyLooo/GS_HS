from env.base_sim_env import BaseMujocoGSWorker
import gym
from gym import spaces
import time
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from gym.vector import AsyncVectorEnv
from easydict import EasyDict


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
        
        # 定义 action 和 observation 空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(worker_config.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, worker_config.gs_config.gs_rgb_height, worker_config.gs_config.gs_rgb_width), dtype=np.uint8),
            "joint_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(worker_config.joint_dim,), dtype=np.float32)
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
        obs = self._get_observation()
        return obs, {}
    
    def _set_state(self, qpos, qvel):
        # assert qpos.shape == (6,) and qvel.shape == (6,)
        self.data.qpos[:6] = np.copy(qpos)
        self.data.qvel[:6] = np.copy(qvel)
        # mujoco 仿真向前推进一步
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        # 应用动作
        self._apply_action(action)
        # 提取观测
        obs = self._get_observation()
        obs = {}

        # 计算 reward、done，示例：
        reward = 0.0
        done = False
        info = {}
        truncated = {}

        return obs, reward, done, truncated, info

    def _apply_action(self, action):
        self.worker.data.ctrl[:] = action
        self.worker._phy_simulation_step()

    def _get_observation(self):
        obs = self.worker._get_cur_obs()
        return obs

    def close(self):
        self.worker.stop()



if __name__ == "__main__":
    cfg = EasyDict({
        "mujoco_config": {
            "model_path": "/home/cfy/cfy/gs_hs/model_asserts/mujoco_asserts/piper/scene.xml", 
        },
        "display_data": True,
        "enable_free_camera": True, 
        "action_dim": 6,
        "joint_dim": 6,
        "gs_config": {
            "gs_rgb_width": 640,  # 你需要设置图像宽度
            "gs_rgb_height": 480,  # 你需要设置图像高度
            "gs_model_dict": {
                "background": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/scene/1lou_0527_res.ply",
                "link1": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link1_rot.ply",
                "link2": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link2_rot.ply",
                "link3": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link3_rot.ply",
                "link4": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link4_rot.ply",
                "link5": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link5_rot.ply",
                "link6": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link6_rot.ply",
                "link7": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link7_rot.ply",
                "link8": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link8_rot.ply",
                "desk": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/object/desk/1louzhuozi_res.ply",

                "apple": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply",
                "banana": "/home/cfy/cfy/gs_hs/model_asserts/3dgs_asserts/object/banana/banana_res.ply",
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

    env = MujocoGSGymEnv(cfg)
    while True:
        # 获取动作维度
        action_dim = env.worker.model.nu

        # 获取每个 actuator 的控制范围
        ctrl_range = env.worker.model.actuator_ctrlrange  # shape: (action_dim, 2)

        # 按控制范围随机采样动作
        action = np.random.uniform(low=ctrl_range[:, 0], high=ctrl_range[:, 1])
        env.step(action)