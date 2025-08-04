"""
flip_normals_blender.py

Blender 批处理脚本：导入单个 OBJ，翻转法线，导出带 _flip 后缀的 OBJ。

用法：
    blender --background --python flip_normals_blender.py -- <input.obj> <output_flip.obj>
"""
import bpy, sys, os

# 解析命令行参数
argv = sys.argv
if "--" not in argv:
    raise RuntimeError("Expected script args after '--'")
idx = argv.index("--")
input_path = argv[idx+1]
output_path = argv[idx+2]

# 确保 OBJ IO 插件可用（Blender 4.x 中默认应加载）
try:
    bpy.ops.preferences.addon_enable(module="io_scene_obj")
except Exception:
    pass

# 重置场景
bpy.ops.wm.read_factory_settings(use_empty=True)

# 导入 OBJ
bpy.ops.import_scene.obj(filepath=input_path)

# 对导入的所有对象翻转法线
for obj in bpy.context.selected_objects:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')

# 导出翻转法线后的 OBJ
bpy.ops.export_scene.obj(
    filepath=output_path,
    use_selection=True,
    use_normals=True,
    use_uvs=True,
    use_materials=False
)
print(f"Flipped normals: {input_path} → {output_path}")
