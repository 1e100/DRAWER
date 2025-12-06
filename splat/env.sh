#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

conda create --name drawer_splat -y python=3.8
conda activate drawer_splat
conda config --env --set channel_priority strict

conda install -y \
  -c "nvidia/label/cuda-11.8.0" \
  cuda-toolkit=11.8.0 \
  cuda-nvcc=11.8.89 \
  cuda-runtime=11.8.0

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDA_ARCH_LIST="8.6"
export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
export CMAKE_CUDA_ARCHITECTURES="$(echo "${CUDA_ARCH_LIST}" | tr ';' ' ' | tr -d '.')"

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 matplotlib --extra-index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -e .
pip install functorch --no-deps
pip install torchmetrics[image]
pip install torchtyping

pip install "typeguard==2.12.1"
pip install --upgrade tyro


pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install accelerate==0.31.0
pip install diffusers==0.29.1
pip install tokenizers==0.15.2
pip install transformers==4.37.2
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 --no-deps

pip install pytorch-lightning==1.9.1
pip install git+https://github.com/xiahongchi/PyWavefront.git
pip install transformations
pip install einops
pip install omegaconf
pip install extcolors
pip install Pylette

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip

pip install kornia==0.7.2
pip install sentence-transformers==2.2.2
pip install albumentations==0.4.3
pip install huggingface-hub==0.24.5

pip install imageio[ffmpeg]

pip install objaverse
