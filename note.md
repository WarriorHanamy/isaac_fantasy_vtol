# 复现笔记

xhost +local:root

```sh
python3 scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096 env.actions.control_action.use_motor_model=False
```
```sh
python3 scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 --enable_fpv_camera
```

To see justfile to run.


<!-- # 测试单机体、开启FPV并录制视频，录制的 fpv_XXX.mp4 会写在当前工作目录
python scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 \
  --checkpoint /workspace/isaac_drone_racer/logs/skrl/drone_racer/2025-10-11_15-54-52_ppo_torch/checkpoints/best_agent.pt \
  env.enable_fpv_camera=True
``` -->
key