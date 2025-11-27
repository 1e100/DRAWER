# CUDA-enabled Isaac Sim environment based on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/isaac_sim

# System deps for Python, Git, and build tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python-is-python3 \
        python3-pip \
        python3-venv \
        git \
        ca-certificates \
        ncurses-term && \
    rm -rf /var/lib/apt/lists/*

# Python deps (no venv/uv inside container)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        "isaacsim[all]==4.5.0" \
        "isaacsim[extscache]==4.5.0" \
        trimesh \
        transformations \
        requests \
        --extra-index-url https://pypi.nvidia.com

# Isaac Lab setup
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    ./isaaclab.sh -i
