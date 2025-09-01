# Direct3D-S2 Docker Environment
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

LABEL name="direct3d-s2" maintainer="direct3d-s2"

# Create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Update package lists and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    wget \
    vim \
    unzip \
    pkg-config \
    cmake \
    curl \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils-extra \
    libeigen3-dev \
    python3-dev \
    python3-setuptools \
    libcgal-dev \
    libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# Limit parallel compilation to avoid OOM
ENV MAX_JOBS=2

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create and activate conda environment
RUN conda create -n direct3d-s2 python=3.10 && echo "source activate direct3d-s2" > ~/.bashrc
ENV PATH /workspace/miniconda3/envs/direct3d-s2/bin:$PATH

# Set conda to always auto-approve
RUN conda config --set always_yes true

# Install conda packages
RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.1.0 -y
RUN conda install -c conda-forge libstdcxx-ng -y

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.5.1 torchvision==0.20.1 xformers --index-url https://download.pytorch.org/whl/cu121

# Clone Direct3D-S2 repository
RUN git clone https://github.com/DreamTechAI/Direct3D-S2.git
WORKDIR /workspace/Direct3D-S2

# Install Direct3D-S2 dependencies
RUN pip install -r requirements.txt

# Install additional dependencies
RUN pip install timm kornia gradio

# Try to install torchsparse using the official installer (fallback to manual compilation)
RUN git clone https://github.com/mit-han-lab/torchsparse.git /workspace/torchsparse && cd /workspace/torchsparse && python -m pip install . || echo "❌ TorchSparse installation failed completely. Direct3D-S2 may work with limited functionality."

# Set global library paths to ensure proper linking at runtime
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/direct3d-s2/lib:${LD_LIBRARY_PATH}"

# Activate conda environment by default
RUN echo "conda activate direct3d-s2" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Create test script to check installation
RUN echo '#!/bin/bash\n\
echo "🧪 Testing Direct3D-S2 installation..."\n\
python -c "import torch; print(f\"✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}\")" || echo "❌ PyTorch import failed"\n\
python -c "import direct3d_s2; print(\"✅ Direct3D-S2 imported successfully\")" || echo "❌ Direct3D-S2 import failed"\n\
python -c "import torchsparse; print(\"✅ TorchSparse imported successfully\")" || echo "⚠️  TorchSparse not available - some features may be limited"\n\
python -c "import timm, kornia, gradio; print(\"✅ Additional dependencies imported successfully\")" || echo "❌ Some dependencies missing"\n\
echo "🎉 Installation check complete!"' > /workspace/test_installation.sh && chmod +x /workspace/test_installation.sh

# Create helper script for manual torchsparse compilation if needed
RUN echo '#!/bin/bash\n\
echo "🔧 Manual TorchSparse compilation..."\n\
cd /workspace\n\
rm -rf torchsparse\n\
git clone https://github.com/mit-han-lab/torchsparse.git\n\
cd torchsparse\n\
export MAX_JOBS=1\n\
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"\n\
python setup.py clean --all\n\
python setup.py install' > /workspace/compile_torchsparse.sh && chmod +x /workspace/compile_torchsparse.sh

# Set working directory to Direct3D-S2 project
WORKDIR /workspace/Direct3D-S2

# Set default command
CMD ["/bin/bash", "-c", "echo 'Welcome to Direct3D-S2 Docker environment!' && echo 'Run /workspace/test_installation.sh to check installation status' && echo 'If TorchSparse failed, try /workspace/compile_torchsparse.sh' && /bin/bash"]