#!/usr/bin/env python3
"""
脚本功能：
    - 读取指定的 PLY 文件
    - 列出所有 element（通常有 "vertex", "face"…）
    - 对于每个 element，打印它的属性名称和数据类型
    - 如果包含 "vertex" element，额外打印前 5 条示例数据
用法：
    python check_ply.py --ply path/to/your.ply
"""

import argparse
from plyfile import PlyData

def inspect_ply(ply_path: str):
    # 1) 读取 PLY
    ply = PlyData.read(ply_path)
    
    print(f"Loaded PLY: {ply_path}\n")
    print("=== Elements and Properties ===")
    
    # 2) 遍历所有 elements
    elem_names = []
    for elem in ply.elements:
        name = elem.name
        elem_names.append(name)
        count = elem.count
        fields = elem.data.dtype.names  # tuple of field names
        
        print(f"- Element '{name}' (count = {count}):")
        for field in fields:
            # 从 dtype.fields 取类型
            field_type = elem.data.dtype.fields[field][0]
            print(f"    • {field} : {field_type}")
        print()
    
    # 3) 如果包含 'vertex' element，则展示前几行数据
    if "vertex" in elem_names:
        vdata = ply["vertex"].data
        n_show = min(5, len(vdata))
        print(f"=== 前 {n_show} 个 vertex 数据示例 ===")
        for i in range(n_show):
            rec = vdata[i]
            line = ", ".join(f"{f}={rec[f]}" for f in vdata.dtype.names)
            print(f"  [{i}] {line}")
    else:
        print("No 'vertex' element found.")

def main():
    parser = argparse.ArgumentParser(description="Inspect PLY elements & properties")
    parser.add_argument("--ply", "-p", required=True, help="Path to .ply file")
    args = parser.parse_args()
    inspect_ply(args.ply)

if __name__ == "__main__":
    main()
