# Start with the NVIDIA CUDA base image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    freeglut3-dev \
    libglib2.0-0 \
    libxcb-util1 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Qt
ENV QT_DEBUG_PLUGINS=1
ENV QT_QPA_PLATFORM=xcb

# Install miniconda
ENV CONDA_DIR="/opt/conda"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash 

# Create a new conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN /opt/conda/bin/conda create -y -n phystwin_env python=3.12 && conda init

# Set the working directory, non-root user, and permissions
WORKDIR /PhysTwin

# CUDA architecture settings
# This is set to 12.0 for NVIDIA RTX 50 series GPUs (Ada Lovelace architecture)
# If you are using a different GPU, make sure to set this to the correct architecture
# You can find the list of CUDA architectures here: https://developer.nvidia.com/cuda-gpus
ARG TORCH_CUDA_ARCH_LIST="12.0+PTX"

# Copy only build-required files (dependencies will be mounted at runtime)
COPY --chmod=755 env_install/ env_install/
COPY --chmod=755 gaussian_splatting/submodules/ gaussian_splatting/submodules/
COPY --chmod=755 data_process/ data_process/

# Activate the conda environment and install dependencies
RUN conda run -n phystwin_env pip install numpy==1.26.4 warp-lang usd-core matplotlib "pyglet<2" open3d trimesh rtree pyrender
RUN conda run -n phystwin_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN conda run -n phystwin_env pip install stannum termcolor fvcore wandb moviepy imageio opencv-python cma Cython pyrealsense2 atomics pynput plyfile einops

# Install the env for grounded-sam-2
RUN conda run -n phystwin_env bash -c "git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && \
    cd Grounded-SAM-2/checkpoints/ && bash download_ckpts.sh && \
    cd ../gdino_checkpoints/ && bash download_ckpts.sh && \
    cd .. && pip install -e . && \
    pip install --no-build-isolation -e grounding_dino"

# Install the env for image upscaler using SDXL
RUN conda run -n phystwin_env pip install diffusers accelerate gsplat==1.4.0 kornia

RUN conda run -n phystwin_env pip install --no-build-isolation gaussian_splatting/submodules/diff-gaussian-rasterization/ && conda run -n phystwin_env pip install --no-build-isolation gaussian_splatting/submodules/simple-knn/

RUN git clone https://github.com/facebookresearch/pytorch3d.git && conda run -n phystwin_env pip install --no-build-isolation pytorch3d/

# Install the env for trellis
# RUN conda run -n phystwin_env cd data_process && git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git && \
#     cd TRELLIS && . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

RUN MAX_JOBS=8 conda run -n phystwin_env pip install flash_attn --no-build-isolation
# Set the default command
CMD ["/bin/bash"]