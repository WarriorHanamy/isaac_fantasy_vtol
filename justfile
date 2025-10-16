
build:
    docker build -f docker/simulation.dockerfile \
    --secret id=gitconfig,src=$HOME/.gitconfig \ ISAACSIM_VERSION
    --build-arg ISAACSIM_VERSION=4.5.0 \
    --build-arg ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git \
    --build-arg ISAACLAB_REF=v2.1.0 \
    --build-arg PROXY=http://127.0.0.1:7890 \
    --network=host --progress=plain \
    -t isaaclab_image:v0 .


run:
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
    isaaclab_image:v0

exec:
    docker exec -it woneuver /bin/bash


sim:
    export DISPLAY=:0
    /workspace/isaaclab/_isaac_sim/isaac-sim.sh --reset-user 