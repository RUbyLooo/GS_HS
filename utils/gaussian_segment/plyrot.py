import open3d as o3d
import numpy as np
import time
from plyfile import PlyData

def load_colored_pointcloud(ply_path):
    ply = PlyData.read(ply_path)
    vertex = ply['vertex'].data

    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

    # 自动判断颜色字段（支持 float or uchar）
    if 'red' in vertex.dtype.names and 'green' in vertex.dtype.names and 'blue' in vertex.dtype.names:
        red = vertex['red']
        green = vertex['green']
        blue = vertex['blue']
        if red.dtype == np.uint8:
            rgb = np.stack([red, green, blue], axis=-1) / 255.0
        else:
            rgb = np.stack([red, green, blue], axis=-1)  # assume already 0~1
    else:
        print("⚠️ 没有找到 red/green/blue 字段，使用默认灰色")
        rgb = np.ones_like(xyz) * 0.5

    # 构建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

# 读取 PLY 点云（替换为你的路径）
ply_path = "/home/jiziheng/Music/robot/gs_scene/mobile_rl/mobile_ai_gs/models/3dgs/robot/quadrotor_res.ply"
pcd = load_colored_pointcloud(ply_path)

# 打印状态
if not pcd.has_colors():
    print("⚠️ 点云没有颜色")
else:
    print("✅ 点云颜色已加载")

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='3DGS Rotation', width=800, height=600)
vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
vis.get_render_option().point_size = 3.0
vis.add_geometry(pcd)

# 固定视角（可调）
ctr = vis.get_view_control()
cam_params = ctr.convert_to_pinhole_camera_parameters()
cam_params.extrinsic = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -2],
    [0, 0, 1, 2],
    [0, 0, 0, 1]
])
ctr.convert_from_pinhole_camera_parameters(cam_params)

# 绕自身 Z 轴旋转动画
for i in range(360):
    R = pcd.get_rotation_matrix_from_axis_angle([0, 0, np.pi/180])
    pcd.rotate(R, center=pcd.get_center())
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.02)

vis.destroy_window()
