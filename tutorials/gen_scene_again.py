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
    # 生成一个带碰撞体和刚体属性的绿色圆锥体 (初始半径 0.1)
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.1,  # <---- 初始半径
        height=1.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-1.0, 0.0, 5.0))

    # 从USD文件向场景中生成一个桌子
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/Objects/Table", cfg, translation=(0.0, 0.0, 1.05))


def main():
    """主函数"""

    # 初始化仿真上下文
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # 设置主相机视角
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # 第一次设计场景 (使用 radius = 0.1)
    design_scene()

    # 第一次重置，加载初始场景到 PhysX
    sim.reset()

    # 准备就绪！
    print("[INFO]: Setup complete... Simulating with radius 0.1. (0-1000 steps)")

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

            # ------------------- [ 已修改的代码块开始 ] -------------------
            
            # 2. 获取 visual 和 collision prims
            # 根据你的最新截图，真正的几何体在 "/World/Objects/ConeRigid/geometry/mesh"
            prim_path = "/World/Objects/ConeRigid/geometry/mesh" # <--- 最终修复
            print(f"[INFO]: Attempting to find prim at: {prim_path}")

            # 查找 "mesh" 图元 (类型是 Cone)
            cone_mesh_prim = prim_utils.get_prim_at_path(prim_path)

            new_radius = 1.0

            # 3. 检查 prims 是否存在并修改属性
            if cone_mesh_prim:
                print(f"[INFO]: Modifying prim attributes to radius {new_radius}...")
                
                # 检查 "radius" 属性是否存在
                if cone_mesh_prim.HasAttribute("radius"):
                    # 修改 'mesh' prim (Cone) 的半径
                    cone_mesh_prim.GetAttribute("radius").Set(new_radius)
                    print("[INFO]: Attributes changed.")
                    
                    # 假设碰撞也由这个 mesh 控制 (这是 Isaac Lab 的常见做法)
                    # 如果碰撞是分开的 (例如 .../geometry/collision)，你可能需要单独修改它
                    # 但对于这个版本的 API，很可能就是修改 mesh 本身。
                
                else:
                    print(f"[ERROR]: Prim at '{prim_path}' does not have a 'radius' attribute. Aborting change.")
                    radius_changed = True
                    sim.play()
                    continue

            else:
                # 如果连 "mesh" 都找不到
                print(f"[ERROR]: Could not find prim at '{prim_path}'. Aborting change.")
                radius_changed = True
                exit()  # 跳出循环，避免无限尝试

            # ------------------- [ 已修改的代码块结束 ] -------------------

            # 4. 【关键步骤】重置模拟器
            #    这将强制 SimulationContext 重新读取 prim 属性
            #    并使用新半径重建 PhysX 几何体。
            print("[INFO]: Calling sim.reset() to reload physics scene...")
            sim.reset()
            print("[INFO]: sim.reset() complete.")

            # 5. 更新标志位
            radius_changed = True

            # 6. 恢复模拟
            print("[INFO]: Resuming simulation with new, larger cone.")
            print("[INFO]: ===========================================\n")
            sim.play()

        # 仅在模拟播放时执行step
        if sim.is_playing():
            sim.step()
            step_count += 1
        else:
            # 如果模拟未运行 (例如在暂停/重置期间)，
            # 必须手动更新 app 以保持窗口响应
            simulation_app.update()


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()