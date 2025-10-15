# syntax=docker/dockerfile:1.4

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=4.5.0

FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS simulation

ARG ISAACSIM_ROOT_PATH=/isaac-sim
ARG ISAACLAB_PATH=/workspace/isaaclab
ARG DOCKER_USER_HOME=/root
ARG ROS2_APT_PACKAGE=ros-base
ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_REF=v2.1.0

ENV ISAACSIM_VERSION=${ISAACSIM_VERSION} \
    ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    DOCKER_USER_HOME=${DOCKER_USER_HOME} \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTRTPS_DEFAULT_PROFILES_FILE=${DOCKER_USER_HOME}/.ros/fastdds.xml \
    CYCLONEDDS_URI=${DOCKER_USER_HOME}/.ros/cyclonedds.xml \
    ISAACSIM_PATH=${ISAACLAB_PATH}/_isaac_sim \
    OMNI_KIT_ALLOW_ROOT=1 \
    http_proxy=http://127.0.0.1:8889 \
    https_proxy=http://127.0.0.1:8889 \
    HTTP_PROXY=http://127.0.0.1:8889 \
    HTTPS_PROXY=http://127.0.0.1:8889 \
    no_proxy=localhost,127.0.0.1,::1 \
    NO_PROXY=localhost,127.0.0.1,::1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

ENV ROS2_APT_PACKAGE=${ROS2_APT_PACKAGE}

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget \
    curl \
    gedit \
    tmux \
    software-properties-common 

RUN git clone --branch ${ISAACLAB_REF} --depth 1 ${ISAACLAB_REPO} ${ISAACLAB_PATH} && \
    rm -rf ${ISAACLAB_PATH}/.git

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh

RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml SciencePlots pxr pytest lark

RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source

RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "numpy<2" --upgrade --no-cache-dir

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo jammy) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-${ROS2_APT_PACKAGE} \
    ros-humble-vision-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rmw-fastrtps-cpp \
    ros-dev-tools && \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py rosdep ${ISAACLAB_PATH}/source && \
    echo "source /opt/ros/humble/setup.bash" >> ${HOME}/.bashrc

RUN mkdir -p ${DOCKER_USER_HOME}/.ros && \
    cp -r ${ISAACLAB_PATH}/docker/.ros/. ${DOCKER_USER_HOME}/.ros/

RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /root/.bashrc

VOLUME [ \
    "/isaac-sim/kit/cache", \
    "/root/.cache/ov", \
    "/root/.cache/pip", \
    "/root/.cache/nvidia/GLCache", \
    "/root/.nv/ComputeCache", \
    "/root/.nvidia-omniverse/logs", \
    "/root/.local/share/ov/data", \
    "/workspace/isaaclab/docs/_build", \
    "/workspace/isaaclab/logs", \
    "/workspace/isaaclab/data_storage" \
]

WORKDIR /workspace
# -----------------------------------------------------
# TODO： 减小本地化size
# RUN apt -y autoremove && apt clean autoclean && \
#     rm -rf /var/lib/apt/lists/*
# TODO： 如果走了代理、但是想镜像本地化到其它机器，记得清空代理（或者容器内unset）
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=

ENTRYPOINT ["/bin/bash"]
