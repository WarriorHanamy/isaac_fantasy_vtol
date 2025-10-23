# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
  ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# 创建参数解析器
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# 添加AppLauncher的命令行参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()
# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
# 导入 PXR 库 (UsdGeom, UsdPhysics)
from pxr import UsdGeom, UsdPhysics, Gf

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    """通过生成地面、灯光、物体和从USD文件加载的网格来设计场景。"""
    # 地面
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # 生成平行光
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # 为所有要生成的对象创建一个新的xform图元作为父级
    prim_utils.create_prim("/World/Objects", "Xform")
    
    # --- [ 已修改的 design_scene 逻辑开始 ] ---
    # 我们将手动创建一个刚体，以解耦视觉和碰撞

    prim_path = "/World/Objects/ConeRigid"
    translation = (-1.0, 0.0, 5.0)
    orientation = (0.0, 0.0, 0.0, 0.0)
    
    # 1. 定义你想要的半径
    initial_visual_radius = 1.0
    initial_collision_radius = 0.1  # <--- 在这里设置你想要的更小的初始碰撞半径

    # 2. 创建父级 Xform (刚体)
    prim = prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    
    # 应用 Rigid Body API
    ##### IMPORTANT
    UsdPhysics.RigidBodyAPI.Apply(prim)
    
    # 应用 Mass API 并设置质量
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(1.0) 
    
    # 3. 创建 VISUAL (视觉) 图元
    
    # --- [ 修复开始 ] ---
    visual_prim_path = f"{prim_path}/visual" # <--- 1. 定义路径
    cfg_visual_cone = sim_utils.ConeCfg(
        # prim_path 不在这里
        radius=initial_visual_radius,
        height=1.0,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False) # <--- 禁用碰撞
    )
    # 3. 在 .func() 中传入路径
    cfg_visual_cone.func(visual_prim_path, cfg_visual_cone) 
    # --- [ 修复结束 ] ---

    # 4. 创建 COLLISION (碰撞) 图元
    
    # --- [ 修复开始 ] ---
    collision_prim_path = f"{prim_path}/collision" # <--- 1. 定义路径
    cfg_collision_cyl = sim_utils.CylinderCfg(
        # prim_path 不在这里
        radius=initial_collision_radius,   # <--- 使用你想要的更小的半径
        height=1.0,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True), # <--- 启用碰撞
        visual_material=None # <--- 没有视觉材质
    )
    # 3. 在 .func() 中传入路径
    cfg_collision_cyl.func(collision_prim_path, cfg_collision_cyl)
    # --- [ 修复结束 ] ---
    
    # 5. 确保碰撞体不可见
    coll_prim = prim_utils.get_prim_at_path(collision_prim_path) # <--- 现在这个路径是正确的
    if coll_prim:
        # 使用 PXR API 使其不可见
        UsdGeom.Imageable(coll_prim).MakeInvisible()
    
    # --- [ 已修改的 design_scene 逻辑结束 ] ---

    # 从USD文件向场景中生成一个桌子
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    """主函数 (此函数无需更改)"""

    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # 第一次设计场景
    design_scene()

    # 第一次重置，加载初始场景到 PhysX
    sim.reset()

    # 准备就绪！
    print("[INFO]: Setup complete... Simulating with decoupled radius. (0-1000 steps)")

    # 模拟物理过程
    step_count = 0
    radius_changed = False  # 跟踪我们是否已经改变了半径

    # 启动模拟
    sim.play()

    # 模拟循环
    while simulation_app.is_running():
        
        # 检查是否需要修改
        if step_count == 1000 and not radius_changed:

            print("\n[INFO]: === Step 1000 Reached: Modifying Scene ===")
            
            # 1. 暂停模拟
            print("[INFO]: Pausing simulation...")
            sim.pause()
            
            visual_prim_path = "/World/Objects/ConeRigid/visual"
            collision_prim_path = "/World/Objects/ConeRigid/collision"

            print(f"[INFO]: Attempting to find visual prim at: {visual_prim_path}")
            print(f"[INFO]: Attempting to find collision prim at: {collision_prim_path}")

            # 查找图元
            # REC： 这里有一个问题，在于必须正确访问到含有 radius 属性的图元(Prim)
            # collision_prim.GetAttribute("radius").等API不能递归地访问到子图元的属性
            visual_prim_path = visual_prim_path + "/geometry/mesh"  # <--- 最终修复
            collision_prim_path = collision_prim_path + "/geometry/mesh"  # <
            visual_prim = prim_utils.get_prim_at_path(visual_prim_path)
            collision_prim = prim_utils.get_prim_at_path(collision_prim_path)

            # 在第1000帧，我们将把它们都设置为 1.0
            new_visual_radius = 1.0
            new_collision_radius = 10.0  # 你也可以把它们设置为不同的值
            
            # 3. 检查 prims 是否存在并修改属性
            if visual_prim and collision_prim:
                print(f"[INFO]: Modifying prim attributes...")
                
                # 3a. 修改 visual prim (Cone)
                if visual_prim.HasAttribute("radius"):
                    visual_prim.GetAttribute("radius").Set(new_visual_radius)
                    print(f"[INFO]: Visual 'radius' set to {new_visual_radius}.")
                else:
                    print(f"[WARN]: Visual prim at '{visual_prim_path}' has no 'radius' attribute.")

                # 3b. 修改 collision prim (Cylinder)
                if collision_prim.HasAttribute("radius"):
                    collision_prim.GetAttribute("radius").Set(new_collision_radius)
                    print(f"[INFO]: Collision 'radius' set to {new_collision_radius}.")
                else:
                    print(f"[WARN]: Collision prim at '{collision_prim_path}' has no 'radius' attribute.")
                
                print("[INFO]: Attributes changed.")

            else:
                # 打印出具体是哪个 prim 没找到
                if not visual_prim:
                    print(f"[ERROR]: Could not find VISUAL prim at '{visual_prim_path}'.")
                if not collision_prim:
                    print(f"[ERROR]: Could not find COLLISION prim at '{collision_prim_path}'.")
                
                print("[ERROR]: Aborting change.")
                radius_changed = True
                sim.play()
                continue # 跳过此循环的其余部分
            
            # 4. 【关键步骤】重置模拟器
            print("[INFO]: Calling sim.reset() to reload physics scene...")
            sim.reset()
            print("[INFO]: sim.reset() complete.")
            radius_changed = True

            print("[INFO]: Resuming simulation...")
            sim.play()

        if sim.is_playing():
            step_count += 1
            sim.step()
        else:
            simulation_app.update()

if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()