
###  Host Operations  ###
build:
    docker build -f docker/simulation.dockerfile \
    --secret id=gitconfig,src=$HOME/.gitconfig \
    --build-arg ISAACSIM_VERSION=4.5.0 \
    --build-arg ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git \
    --build-arg ISAACLAB_REF=v2.1.0 \
    --network=host --progress=plain \
    -t vtol_rl:v0 .


run-container:
    xhost +local:docker
    docker run --name woneuver -itd --privileged --gpus all --network host \
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
    -v $(pwd):/workspace/isaac_drone_racer \
    vtol_rl:v0

run: run-container
r: run-container

exec:
    docker exec -it woneuver /bin/bash

stop:
    docker stop woneuver && docker rm woneuver

s: stop

###  Container Operations  ###
init:
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e . && \
    export DISPLAY=:0 
    /workspace/isaaclab/_isaac_sim/isaac-sim.sh --reset-user \
sim:
    export DISPLAY=:0 
    /workspace/isaaclab/_isaac_sim/isaac-sim.sh --reset-user \

train:
    ${ISAACLAB_PATH}/_isaac_sim/python.sh scripts/rl/train.py \
    --task Isaac-Drone-Racer-v0 --headless --num_envs 4096

demo:
    ${ISAACLAB_PATH}/_isaac_sim/python.sh \
    scripts/rl/play.py --task Isaac-Drone-Racer-Play-v0 --num_envs 1 --device cpu


### tutorials for totally beginners
try:
    ${ISAACLAB_PATH}/_isaac_sim/python.sh \
    tutorials/generate_scene.py --device cpu

compare-reward:
    export DISPLAY=:0 
    ${ISAACLAB_PATH}/_isaac_sim/python.sh \
    tutorials/shaped_reward_comparison.py --device cpu


### Records
record-video:
    ffmpeg -video_size 2560x1440 -framerate 25 -f x11grab \
    -i :0.0 output.mp4


train-logged:
      ${ISAACLAB_PATH}/_isaac_sim/python.sh scripts/rl/train.py \
      --task Isaac-Drone-Racer-v0 --headless --num_envs 4096 \
      agent.experiment.write_interval=1000 \
      trainer.environment_info=log