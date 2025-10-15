# 复现笔记
## 配置
git clone以前apt安装`git-lfs`，或者之后`git lfs pull`拉素材
```sh
docker build -f docker/simulation.dockerfile \
  --build-arg ISAACSIM_VERSION=4.5.0 \
  --build-arg ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git \
  --build-arg ISAACLAB_REF=v2.1.0 \
  --network=host --progress=plain \
  -t isaaclab_image:v0 .

xhost +local:root

docker run --name test-isaaclab -itd --privileged --gpus all --network host \
  --entrypoint bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v /home/dzp/projects/isaac_drone_racer:/workspace/isaac_drone_racer \
  isaaclab_image:v0

docker exec -it test-isaaclab /bin/bash

# 可以测试GUI和RL功能是否正常 
# python isaaclab/scripts/tutorials/00_sim/log_time.py --headless
# python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

容器内干净启动一个isaac-sim GUI
```sh
export DISPLAY=:0

/workspace/isaaclab/_isaac_sim/isaac-sim.sh --reset-user 
```
窗口会在宿主机弹出；打开某些硬编码到`绝对路径`的USD（e.g., `gate.usd`）,对应纹理的路径改成新的绝对路径: `/workspace/isaac_drone_racer/assets/gate/textures/bitmap.png`,然后保存usd即可（不需要保存scene）

运行穿越机本来的功能代码
```sh
cd isaac_drone_racer && ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .

python -m pytest tests/test_dynamics.py

python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096
# export DISPLAY=:0 && python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --num_envs 128
# python scripts/rl/train.py --task Isaac-Drone-Racer-v0 --headless --num_envs 4096 env.actions.control_action.use_motor_model=False

export DISPLAY=:0

# 测试单机体、开启FPV并录制视频，录制的 fpv_XXX.mp4 会写在当前工作目录
python scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 \
  --checkpoint /workspace/isaac_drone_racer/logs/skrl/drone_racer/2025-10-11_15-54-52_ppo_torch/checkpoints/best_agent.pt \
  env.enable_fpv_camera=True
```
