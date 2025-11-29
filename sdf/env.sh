#!/usr/bin/env bash

# set -euo pipefail

REQUIRED_APT_PACKAGES=(
  libavformat-dev
  libavcodec-dev
  libavdevice-dev
  libavutil-dev
  libavfilter-dev
  libswscale-dev
  libswresample-dev
  build-essential
)

# ---- Check each package with dpkg-query ----
for pkg in "${REQUIRED_APT_PACKAGES[@]}"; do
  # dpkg-query returns non-zero if not installed; we invert it with !
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    missing_packages+=("$pkg")
  fi
done

# ---- If any are missing, print and exit with error ----
if [ "${#missing_packages[@]}" -ne 0 ]; then
  echo "Error: the following APT packages are missing:" >&2
  for pkg in "${missing_packages[@]}"; do
    echo "  - $pkg" >&2
  done
  echo "Install them with something like:" >&2
  echo "  sudo apt-get update && sudo apt-get install ${missing_packages[*]}" >&2
  exit 1
fi

# Check that gcc-11 is installed
if ! command -v gcc-11 >/dev/null 2>&1; then
  echo "Error: gcc-11 (GCC 11) is not installed or not in PATH." >&2
  exit 1
fi

# Check that g++-11 is installed
if ! command -v g++-11 >/dev/null 2>&1; then
  echo "Error: g++-11 (G++ 11) is not installed or not in PATH." >&2
  exit 1
fi

echo "Binary dependency check was successful."

eval "$(conda shell.bash hook)"

conda create --name drawer_sdf -y python=3.8
conda activate drawer_sdf

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

# Make sure Cython is compatible with PyAV 9.2
pip install "cython<3"
conda install -y -c conda-forge "ffmpeg<5" "av=9.2.0"

# ensure gcc version 11.x and nvcc version 11.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 matplotlib --extra-index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

PIP_CONSTRAINT=sdf/pip_constraints.txt pip install -e sdf
pip install functorch --no-deps
pip install torchmetrics[image]
pip install torchtyping

pip install accelerate==0.27.2
pip install diffusers==0.30.2
pip install tokenizers==0.15.2
pip install transformers==4.37.2
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 --no-deps
pip install omegaconf
pip install tabulate
pip install pandas

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

pip install "typeguard==2.12.1"
pip install --upgrade tyro

pip install git+https://github.com/NVlabs/nvdiffrast.git

pip install scikit-learn
pip install imageio[ffmpeg]

pip install hydra-core --upgrade --pre
pip install hydra-submitit-launcher --upgrade
pip install visdom

pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install transformations

