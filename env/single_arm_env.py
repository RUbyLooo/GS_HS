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
from scipy.spatial.transform import Rotation as R
import copy
import pygpg
from mujoco import MjModel, MjData, mjtObj
from termcolor import cprint
import threading



class SingleArmEnv(BaseEnv):
    def __init__(self, path, cfg):
        super().__init__(path, cfg)
        self.path = path
        self.latest_images = None
        self.render_lock = threading.Lock()
        for i in range(1,9):
            self.add_gaussian_model(f"link{i}",f"/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/robot/piper/arm_link{i}_rot.ply")
        self.add_gaussian_model(
            "apple", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/apple/apple_res.ply"
        )
        self.add_gaussian_model(
            "banana", "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/3dgs_asserts/object/banana/banana_res.ply"
        )
        # print(self.gs_model_dict)
        # self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        # self.render_thread.start()
        self.action = None

    def unnormalizar_gripper(self,gripper_action ):
        return gripper_action * 0.035
    

    def run_func(self):
        # self.show_gs_render()
        # self.action[0] = np.array([1,1,1,1,1,0,1])
        self.data.qpos[:6] = self.action[0][:6]
        self.data.qpos[6] = self.unnormalizar_gripper(self.action[0][6])
        self.data.qpos[7] = -self.unnormalizar_gripper(self.action[0][6])
        mujoco.mj_step(self.model, self.data)
        self.sync()
        # print(f"in run func action is : {self.action}")
        self.update_gs_scene()

    def run_before(self):
        self.update_gs_scene()
        pass
    

    # def render_loop(self):
    #     self.update_gs_scene()
    #     print("[render_loop] Scene updated before first frame.")
    #     while True:
    #         self.show_gs_render()
    #         rgb_imgs, _, _, _ = self.get_img()
            # with self.render_lock:
            #     self.latest_images = rgb_imgs
            # time.sleep(1 / 30.)  # 假设你想 30FPS 渲染
    
    def get_img_thread(self):
        with self.render_lock:
            if self.latest_images is not None:
                return self.latest_images, None, None, None
            else:
                # 如果线程还没准备好，直接快速 fallback 一次渲染
                print("[get_img_thread] No image ready, fallback to immediate render.")
                return self.get_img()


def test():
    cfg = {
        "is_have_moving_base": False,
        "is_have_arm": True,
        "is_use_gs_render": True,
        "episode_len": 100,
        "is_save_record_data": True,
        "camera_names": ["3rd_camera", "wrist_cam"],
        "robot_link_list": ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"],
        "env_name": "SingleArmEnv",
        "obj_list": ["desk","apple","banana"]
    }
    env = SingleArmEnv( "/home/jiziheng/Music/robot/gs_scene/gs_hs/model_asserts/mujoco_asserts/piper/scene.xml",cfg)
    env.run_loop()

if __name__ == "__main__":
    test()