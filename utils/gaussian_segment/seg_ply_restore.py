import numpy as np
from plyfile import PlyData, PlyElement
import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import tqdm
import math
from termcolor import cprint
import os
import yaml
import argparse

def restore_segmented_ply_fields(original_ply, segmented_ply, output_ply, scale_factor, tolerance=1e-6):
    # 读取原始点云
    orig = PlyData.read(original_ply)
    orig_verts = orig['vertex'].data
    orig_coords = np.stack([orig_verts['x'], orig_verts['y'], orig_verts['z']], axis=-1)

    # 读取分割后的点云
    seg = PlyData.read(segmented_ply)
    seg_verts = seg['vertex'].data
    seg_coords = np.stack([seg_verts['x'], seg_verts['y'], seg_verts['z']], axis=-1)

    # 构建原始点索引的 k-d tree（加速坐标查找）
    
    tree = cKDTree(orig_coords)
    matched_indices = tree.query(seg_coords, distance_upper_bound=tolerance)[1]

    # 过滤未匹配成功的点
    valid_mask = matched_indices < len(orig_coords)
    matched_indices = matched_indices[valid_mask]
    seg_coords = seg_coords[valid_mask]
    cprint(text = f"valid matched points number: {len(matched_indices)} / {len(seg_verts)}", color="blue",attrs=["bold"])

    # 准备所有字段
    available_fields = orig_verts.dtype.names
    output_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    for name in available_fields:
        if name not in ['x', 'y', 'z']:
            output_dtype.append((name, 'f4'))

    # 构建输出点云
    output_data = []
    for i, idx in enumerate(tqdm.tqdm(matched_indices)):

        vertex = [seg_coords[i][0] * scale_factor , seg_coords[i][1] * scale_factor, seg_coords[i][2] * scale_factor] 
        for name in available_fields:
            if name  in ['x', 'y', 'z']:
                continue
                # vertex.append(orig_verts[name][idx])
            elif name in ['scale_0', 'scale_1', 'scale_2']:
                vertex.append(orig_verts[name][idx] + math.log(scale_factor) ) ##缩放scale
            else:
                vertex.append(orig_verts[name][idx])
        output_data.append(tuple(vertex))

    output_data_np = np.array(output_data, dtype=output_dtype)
    el = PlyElement.describe(output_data_np, 'vertex')
    PlyData([el], text=False).write(output_ply)


def transform_ply_points(input_ply, output_ply, transform_matrix):
    ply = PlyData.read(input_ply)
    verts = ply['vertex'].data

    coords = np.stack([verts['x'], verts['y'], verts['z']], axis=-1)
    coords_homo = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    transformed = (transform_matrix @ coords_homo.T).T[:, :3]

    R_transform = transform_matrix[:3, :3]  # 提取旋转矩阵

    new_data = []
    dtype = []
    has_rot = all(f'rot_{i}' in verts.dtype.names for i in range(4))
    for name in verts.dtype.names:
        if name in ['x', 'y', 'z']:
            continue
        dtype.append((name, verts.dtype[name]))

    full_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + dtype
    for i in tqdm.tqdm(range(len(verts))):
        fields = [transformed[i, 0], transformed[i, 1], transformed[i, 2]]
        for name in verts.dtype.names:
            if name in ['x', 'y', 'z']:
                continue
            elif name in ['rot_0', 'rot_1','rot_2','rot_3'] and has_rot:
                continue
            fields.append(verts[name][i])
        
        if has_rot:
            # quat_orig = np.array([verts[f'rot_{j}'][i] for j in range(4)])
            quat_orig = np.array([verts['rot_1'][i], verts['rot_2'][i], verts['rot_3'][i], verts['rot_0'][i]])
            rot_orig = R.from_quat(quat_orig)
            rot_new = R.from_matrix(R_transform @ rot_orig.as_matrix())
            quat_new = rot_new.as_quat()
            quat_new = quat_new / np.linalg.norm(quat_new)

            # fields.extend(quat_new.tolist())
            fields.extend([quat_new[3], quat_new[0], quat_new[1], quat_new[2]])

        new_data.append(tuple(fields))

    vertex_array = np.array(new_data, dtype=full_dtype)
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el], text=False).write(output_ply)

    return output_ply


def parse_matrix_from_text(text: str) -> np.ndarray:
    """
    将 4x4 矩阵的字符串（按行）解析为 np.array 格式。
    输入：
        text: 字符串，每行一个 4 元素，空格分隔
    输出：
        np.array(shape=(4,4), dtype=np.float32)
    """
    lines = text.strip().split('\n')
    matrix = []
    for line in lines:
        numbers = [float(num) for num in line.strip().split()]
        matrix.append(numbers)
    return np.array(matrix, dtype=np.float32)   


def merge_gs_pointclouds(source_path,
                        ply1_cfg,
                        ply2_cfg,
                        ):
    """
    将两份带有 GS 属性（例如 scale_0, scale_1, scale_2, rot_0…rot_3 等）的 PLY
    合并到一个新的 PLY 中。要求两份 PLY 的 vertex dtype 完全一致，否则会报错。

    Args:
        ply_path1: 第一份 PLY 的路径
        ply_path2: 第二份 PLY 的路径
        output_ply: 合并后要写入的 PLY 路径

    Raises:
        FileNotFoundError: 如果某个输入文件不存在
        ValueError: 如果两份 PLY 的 vertex 结构(dtype)不一致
    """

    # ==========cfg============
    source_path = source_path
    ply_1_path = os.path.join(source_path, ply1_cfg['path'])
    apply_trans_1 = ply1_cfg['apply_trans']
    ply_1_matrix = ply1_cfg['matrix']
    ply1_keep_trans = bool(ply1_cfg['keep_transformed'])

    ply_2_path =  os.path.join(source_path, ply2_cfg['path'])
    apply_trans_2 = ply2_cfg['apply_trans']
    ply_2_matrix = ply2_cfg['matrix']  
    ply2_keep_trans = bool(ply2_cfg['keep_transformed'])

    out_path = os.path.splitext(ply_1_path)[0] + "_" + os.path.splitext(ply2_cfg['path'])[0] + "_merged" + os.path.splitext(ply_1_path)[1] 
    # ==========cfg============

    if not os.path.isfile(ply_1_path):
        raise FileNotFoundError(f"[Error] Can not find: {ply_1_path}")
    if apply_trans_1:
        cprint(f"[Processing] Doing transformation for {ply_1_path} ...", color="cyan")
        ply1_trans_path = os.path.splitext(ply_1_path)[0]+"_trans"+os.path.splitext(ply_1_path)[1]
        transform_matrix = parse_matrix_from_text(ply_1_matrix)
        transform_ply_points(
            input_ply=ply_1_path,
            output_ply=ply1_trans_path,
            transform_matrix=transform_matrix,
        )
        ply1 = PlyData.read(ply1_trans_path)
    else:
        cprint(f"[Info] Do not need transformation: {ply_1_path}", color="cyan")
        ply1 = PlyData.read(ply_1_path)
    verts1 = ply1['vertex'].data

    if not os.path.isfile(ply_2_path):
        raise FileNotFoundError(f"[Error] Can not find: {ply_2_path}")
    if apply_trans_2:
        cprint(f"[Processing] Doing transformation for {ply_2_path} ...", color="cyan")
        ply2_trans_path = os.path.splitext(ply_2_path)[0]+"_trans"+os.path.splitext(ply_2_path)[1]
        transform_matrix = parse_matrix_from_text(ply_2_matrix)
        transform_ply_points(
            input_ply=ply_2_path,
            output_ply=ply2_trans_path,
            transform_matrix=transform_matrix,
        )
        ply2 = PlyData.read(ply2_trans_path)
    else:
        cprint(f"[Info] Do not need transformation: {ply_2_path} ", color="cyan")
        ply2 = PlyData.read(ply_2_path)
    ply2 = PlyData.read(ply_2_path)
    verts2 = ply2['vertex'].data

    if verts1.dtype != verts2.dtype:
        raise ValueError(
            "[Error] Cannot merge because: PLY vertex field types are inconsistent. \n"
            f"ply1.dtype = {verts1.dtype}\n"
            f"ply2.dtype = {verts2.dtype}"
        )
    
    merged_verts = np.concatenate([verts1, verts2])

    # 5. 将合并后的数组写入 output_ply
    el = PlyElement.describe(merged_verts, 'vertex')
    PlyData([el], text=False).write(out_path)

    cprint(f"Merge Success!\n  {ply_1_path}\n  {ply_2_path}\n merged to\n  {out_path}\nwith {len(merged_verts)} points", color = "cyan")

    if apply_trans_1 and not ply1_keep_trans:
        if os.path.isfile(ply2_trans_path):
            try:
                os.remove(ply2_trans_path)
                cprint(f"[Info] Delet middle transfomred file: {ply1_trans_path}!", color="cyan", attrs=["bold"])
            except Exception as e:
                cprint(f"[Error] Deleting {ply1_trans_path}:{e}", color="red", attrs=["bold"])


    if apply_trans_2 and not ply2_keep_trans:
        if os.path.isfile(ply2_trans_path):
            try:
                os.remove(ply2_trans_path)
                cprint(f"[Info] Delet middle transfomred file: {ply2_trans_path}!", color="cyan", attrs=["bold"])
            except Exception as e:
                cprint(f"[Error] Deleting {ply2_trans_path}:{e}", color="red", attrs=["bold"])

def main():
    parser = argparse.ArgumentParser(description="Restore/transform PLY files using a YAML config.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to your config.yaml (e.g. utils/gauss/config.yaml)"
    )
    args = parser.parse_args()
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Could not find YAML at {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    source_path   = cfg['source_path']

    NEED_RESTORE = bool(cfg["need_restore"])
    MERGE_FLAG = bool(cfg['need_merge'])

    # ========= For retore ===========
    original_ply  = os.path.join(source_path, cfg['original_ply'])
    segmented_ply = os.path.join(source_path, cfg['segmented_ply'])
    seg_res_ply   = os.path.join(source_path, cfg['seg_res_ply'])
    trans_res_ply = os.path.join(source_path, cfg['trans_res_ply'])
    # ========= For retore ===========

    if NEED_RESTORE:
        cprint("========Doing Restore========", color="cyan")
        RESTORE_SEG   = bool(cfg['restore_seg'])
        RESTORE_TRANS = bool(cfg['restore_trans'])
        
        scale_factor  = float(cfg['scale_factor'])

        transform_matrix = cfg['transform']
        transform = parse_matrix_from_text(transform_matrix)
        assert transform.shape == (4,4), "Transform matrix must be 4×4 in the YAML."

        if RESTORE_SEG:
            cprint("[Processing] Running restore_segmented_ply_fields() ...", color="cyan", attrs=["bold"])
            restore_segmented_ply_fields(
                original_ply=original_ply,
                segmented_ply=segmented_ply,
                output_ply=seg_res_ply,
                scale_factor=scale_factor
            )
            final_ply = seg_res_ply
        else:
            cprint(f"[Info] RESTORE_SEG = {RESTORE_SEG} → skipping segment restoration", color="cyan")
            final_ply = original_ply

        if RESTORE_TRANS:
            cprint("[Processing] Running transform_ply_points() ...", color="cyan", attrs=["bold"])
            # If we just restored segments, input is seg_res_ply; otherwise segmented_ply
            input_for_transform = seg_res_ply if RESTORE_SEG else original_ply
            transform_ply_points(
                input_ply=input_for_transform,
                output_ply=trans_res_ply,
                transform_matrix=transform
            )
            final_ply = trans_res_ply
        else:
            cprint(f"[Info] RESTORE_TRANS = {RESTORE_TRANS} → skipping transformation", color="cyan")
            # If no transform, final_ply stays as whatever was assigned after segment step

        cprint(f"[Info] Finished seg and trans. Final output saved to: {final_ply}", color="cyan", attrs=["bold"])
        cprint("========Down Restore========", color="cyan")

    if MERGE_FLAG:
        cprint("========Doing Merge========", color="cyan")
        ply1_cfg = cfg['ply1']
        ply2_cfg = cfg['ply2']

        merge_gs_pointclouds(
            source_path = source_path,
            ply1_cfg = ply1_cfg,
            ply2_cfg = ply2_cfg,
        )
        cprint("========Down Merge========", color="cyan")
    if (not MERGE_FLAG) and (not NEED_RESTORE):
        cprint("[warning] Do not need restore or merge, exited...", color="red",attrs=["bold"])





if __name__ == "__main__":
    main()

    
