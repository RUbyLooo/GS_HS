#!/usr/bin/env python3
import os
import glob
import sys
import shutil
import subprocess
import xml.etree.ElementTree as ET

def flip_obj_normals(src: str, dst: str):
    """
    通过翻转 vn 和 face 顶点顺序来实现法线翻转，不依赖 Blender。
    """
    with open(src, "r") as f:
        lines = f.readlines()
    out = []
    for L in lines:
        if L.startswith("vn "):
            _, x, y, z = L.split()
            out.append(f"vn {-float(x):.6f} {-float(y):.6f} {-float(z):.6f}\n")
        elif L.startswith("f "):
            parts = L.strip().split()[1:]
            out.append("f " + " ".join(parts[::-1]) + "\n")
        else:
            out.append(L)
    with open(dst, "w") as f:
        f.writelines(out)

def run_flip(obj_dir: str):
    """
    遍历 obj_dir 下所有 .obj 文件，生成 *_flip.obj
    """
    for src in glob.glob(os.path.join(obj_dir, "*.obj")):
        name = os.path.splitext(os.path.basename(src))[0]
        if name.endswith("_flip"):
            continue
        dst = os.path.join(obj_dir, f"{name}_flip.obj")
        if not os.path.exists(dst):
            print(f"[1] Flipping normals: {src} → {dst}")
            flip_obj_normals(src, dst)

def run_obj2mjcf(obj_dir: str):
    """
    调用 obj2mjcf 生成 MJCF，自动回答 'y' 覆盖旧文件。
    """
    exe = shutil.which("obj2mjcf")
    if exe:
        cmd = [exe, "--obj-dir", obj_dir, "--save-mjcf", "--add-free-joint", "--decompose"]
    else:
        cmd = [sys.executable, "-m", "obj2mjcf", "--obj-dir", obj_dir, "--save-mjcf", "--add-free-joint", "--decompose"]
    print(f"[2] Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
    proc.communicate(input="y\n")
    if proc.returncode != 0:
        sys.exit(f"ERROR: obj2mjcf failed (code {proc.returncode})")

def patch_mjcf(obj_dir: str, model_name: str):
    """
    将 obj_dir/model_name 下的 MJCF 补丁：
      - 插入 <visual> 配置头灯、雾化、全局视角
      - 复制 .obj, .mtl, 贴图到 model 目录
      - 清理并重写 <asset> 下的 groundplane、mesh_tex、mesh、mesh_visual、mesh_collision
      - 在 <worldbody> 下添加 floor，再重写 body 下的 geoms
    """
    model_dir = os.path.join(obj_dir, model_name)
    xmls = glob.glob(os.path.join(model_dir, "*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No MJCF XML found in {model_dir}")
    mjcf_path = xmls[0]

    # 复制资源文件到 model_dir
    for pat in ("*.obj", "*_flip.obj", "*.mtl", "*.png", "*.jpg"):
        for src in glob.glob(os.path.join(obj_dir, pat)):
            dst = os.path.join(model_dir, os.path.basename(src))
            if not os.path.exists(dst):
                print(f"[3] Copy {os.path.basename(src)} → {model_dir}")
                shutil.copy(src, dst)

    # 找到第一张贴图文件
    textures = glob.glob(os.path.join(model_dir, "*.png")) + glob.glob(os.path.join(model_dir, "*.jpg"))
    if not textures:
        raise FileNotFoundError(f"No texture image in {model_dir}")
    tex_file = os.path.basename(textures[0])

    print(f"[4] Patching MJCF: {mjcf_path}")
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    root.set("model", model_name)

    # —— 在 <mujoco> 下插入 <visual> 块 —— #
    visual = ET.Element("visual")
    ET.SubElement(visual, "headlight", {
        "diffuse":  "0.6 0.6 0.6",
        "ambient":  "0.3 0.3 0.3",
        "specular": "0 0 0",
    })
    ET.SubElement(visual, "rgba", {
        "haze": "0.15 0.25 0.35 1",
    })
    ET.SubElement(visual, "global", {
        "azimuth":   "120",
        "elevation": "-20",
    })
    root.insert(0, visual)
    # —— 结束 —— #

    # 清理 asset 下的旧标签
    asset = root.find("asset")
    for tag in ("texture", "material", "mesh"):
        for e in asset.findall(tag):
            asset.remove(e)

    # 添加地面纹理 & 材质
    ET.SubElement(asset, "texture", {
        "type":    "2d",
        "name":    "groundplane",
        "builtin": "checker",
        "mark":    "edge",
        "rgb1":    "0.2 0.3 0.4",
        "rgb2":    "0.1 0.2 0.3",
        "markrgb": "0.8 0.8 0.8",
        "width":   "300",
        "height":  "300",
    })
    ET.SubElement(asset, "material", {
        "name":       "groundplane",
        "texture":    "groundplane",
        "texuniform": "true",
        "texrepeat":  "5 5",
        "reflectance":"0.2",
    })

    # 添加模型纹理 & 材质
    ET.SubElement(asset, "texture", {
        "type": "2d",
        "name": "mesh_tex",
        "file": tex_file,
    })
    ET.SubElement(asset, "material", {
        "name":      "mesh",
        "texture":   "mesh_tex",
        "specular":  "0.5",
        "shininess":"0.5",
    })

    # 添加可视化 & 碰撞 mesh
    ET.SubElement(asset, "mesh", {
        "name":  "mesh_visual",
        "file":  f"{model_name}_flip.obj",
        "scale": "1 1 1",
    })
    ET.SubElement(asset, "mesh", {
        "name":  "mesh_collision",
        "file":  f"{model_name}.obj",
        "scale": "1 1 1",
    })

    # 在 worldbody 下补地面 & body 下重写 geom
    wb = root.find("worldbody")
    # floor
    ET.SubElement(wb, "geom", {
        "name":     "floor",
        "type":     "plane",
        "size":     "0 0 0.05",
        "material": "groundplane",
    })
    body = wb.find("body")
    # 删除旧 geom
    for g in body.findall("geom"):
        body.remove(g)
    # 保留 freejoint
    # visual geom
    ET.SubElement(body, "geom", {
        "class":      "visual",
        "type":       "mesh",
        "mesh":       "mesh_visual",
        "contype":    "0",
        "conaffinity":"0",
        "material":   "mesh",
    })
    # collision geom
    ET.SubElement(body, "geom", {
        "class":      "collision",
        "type":       "mesh",
        "mesh":       "mesh_collision",
        "contype":    "1",
        "conaffinity":"1",
    })

    # 写回文件
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    print(f"[5] Done patching {mjcf_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flip normals + obj2mjcf + patch MJCF 全流程")
    parser.add_argument("--obj-dir",   "-d", required=True, help="包含 *.obj 的上层目录")
    parser.add_argument("--model-name","-m", required=True, help="MJCF 中 body/model 的名称")
    args = parser.parse_args()

    run_flip(args.obj_dir)
    run_obj2mjcf(args.obj_dir)
    patch_mjcf(args.obj_dir, args.model_name)
