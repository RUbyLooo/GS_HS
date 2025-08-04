#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys, os, time, yaml
import numpy as np
from env.fix_dual_arm_env import FixDualArmEnv
ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------- 常量 -------
GRIP_IDX_LEFT  = 6          # 夹爪 actuator index
GRIP_IDX_RIGHT  = 13          # 夹爪 actuator index
GRIP_OPEN = 0.035     # 打开
GRIP_CLOSE= 0.000      # 闭合(示例)

def plan_sequence(env, arm_sequence, base_name, q_init):
    """
    根据 arm_sequence 返回:
        traj_list: [dict(path, pre_action, post_action, enable)]
        q_last   : 最后关节角 (6,)
    """
    base_pos, base_ori = env.get_body_pose(base_name)
    euler_default = np.array([0.0, 2.35, 0.0])

    ref_q = q_init.copy()
    traj_list = []

    for step in arm_sequence:
        if step["site"] == "zero_site":
            target_joints_state = np.zeros(6)
            path = env.calc_arm_rrt_cubic_traj(ref_q, target_joints_state)
            traj_list.append(dict(
                path=path,
                pre_action=step["pre_action"].lower(),
                post_action=step["action"].lower(),
                enable=step.get("enable", 0)
            ))
            ref_q = np.asarray(target_joints_state).flatten()
            continue
        print(step["site"])
        tgt_pos, tgt_ori = env.get_site_pos_ori(step["site"])
        # if step["site"] == "banana_top_site":
        #     euler_default = np.array([0.0, 2.6, 0.0])
        path, q_last = env.calc_arm_plan_path(
            tgt_pos, 
            tgt_ori,
            base_pos, 
            base_ori,
            ref_q
        )
        traj_list.append(dict(
            path=path,
            pre_action=step["pre_action"].lower(),
            post_action=step["action"].lower(),
            enable=step.get("enable", 0)
        ))
        ref_q = np.asarray(q_last).flatten()

    return traj_list, ref_q
# ==============================================================
# ------------ Episode 执行 ------------------------------------
# ==============================================================
def play_traj(env, traj_list, sleep_dt):
    """依次执行单臂 traj_list"""
    for seg in traj_list:
        # a) 先设夹爪 pre_action
        pre_gripper_ctrl = GRIP_CLOSE if seg["pre_action"] == "close" else GRIP_OPEN
        gripper_ctrl = GRIP_CLOSE if seg["post_action"] == "close" else GRIP_OPEN
        print(f"pre_gripper_ctrl : {pre_gripper_ctrl}")
        print(f"gripper_ctrl : {gripper_ctrl}")

    

        # b) 播放关节轨迹 —— 注意转置
        path = seg["path"]
        for q in path.T:  # shape (6, T) → T 个 6 维点
            env.data.ctrl[:6] = q
            env.data.ctrl[GRIP_IDX_LEFT] = pre_gripper_ctrl
            env.step(env.data.ctrl)
            if sleep_dt:
                time.sleep(sleep_dt)

        # c) enable 等待
        if seg["enable"] > 0:
            for _ in range(seg["enable"]):
                env.data.ctrl[GRIP_IDX_LEFT] = gripper_ctrl
                env.step(env.data.ctrl)
                if sleep_dt:
                    time.sleep(sleep_dt)


def run_episode(env, cfg):
    """一次完整任务：依次执行所有 sub_tasks"""
    env.reset_model()

    base_name  = cfg["left_base_link"]
    q_curr     = np.array(cfg["init_q_left"])

    for sub in cfg["sub_tasks"]:
        seq = sub["arm_sequence"]
        if not seq:                       # 没有臂动作直接跳过
            continue

        traj, q_curr = plan_sequence(env, seq, base_name, q_curr)
        # print(traj)
        play_traj(env, traj, cfg["sleep_dt"])

    return True
# ==============================================================
# --------------------------- 主流程 ----------------------------
# ==============================================================
def main():
    YAML_PATH = "./config/task/record_subtask_dual.yaml"
    with open(YAML_PATH, 'r') as f:
        task_cfg = yaml.safe_load(f)
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
        "obj_list": ["mobile_ai","desk","apple","banana"],
        "left_base_link": "left_base_link",
        "right_base_link": "right_base_link",
        "init_q_left":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "init_q_right":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    }
    cfg.update(task_cfg)
    env = FixDualArmEnv("model_asserts/mujoco_asserts/mobile_ai_robot/scene.xml", cfg)
    num_eps = cfg["episode_len"]
    try:
        for ep in range(num_eps):
            print(f"\n===== Episode {ep+1}/{num_eps} =====")
            run_episode(env, cfg)
    finally:
        env.close()


if __name__ == "__main__":
    main()