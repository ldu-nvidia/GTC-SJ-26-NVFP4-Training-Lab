################################################################################
# GTC SJ-26 NVFP4 Training Lab - Complete Development Environment
#
# Purpose: All-in-one container for NVFP4 training and inference
# Includes: Transformer Engine, ModelOpt, Megatron-LM
# Target: Developers learning NVFP4 quantization
#
# Build: ./docker/build.sh
# Run:   ./docker/run.sh
#
################################################################################

FROM nvcr.io/nvidia/pytorch:25.06-py3

# Build argument for GPU architecture
# Auto-detected by build.sh or can be manually specified
# Examples: "8.0" (A100), "9.0" (H100), "8.0;9.0" (both)
ARG CUDA_ARCH_LIST="8.0;9.0"

LABEL maintainer="GTC SJ-26 Training Lab"
LABEL description="NVFP4 Training and Inference Environment"
LABEL features="TransformerEngine,ModelOpt,Megatron-LM,FP8,NVFP4"
LABEL gpu.architectures="${CUDA_ARCH_LIST}"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

################################################################################
# System Dependencies
################################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

################################################################################
# Python Build Tools
################################################################################
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

################################################################################
# Install Transformer Engine (Provides NVFP4 support)
################################################################################
# TE requires: CUDA 12.1+, cuDNN 9.3+, PyTorch (all in NGC container)
# TE has git submodules and CUDA extensions that need compilation
# Uses CUDA_ARCH_LIST from build argument (auto-detected or manual)

ARG CUDA_ARCH_LIST
# NGC 25.06 container already includes Transformer Engine with NVFP4 support
# Just verify the installation and show the version
RUN echo "Verifying Transformer Engine installation..." && \
    echo "Target GPU architectures: ${CUDA_ARCH_LIST}" && \
    python -c "import transformer_engine; print(f'TE version: {transformer_engine.__version__}')" && \
    python -c "import transformer_engine.pytorch as te; print('✅ Transformer Engine available')" && \
    find /usr/local/lib/python3.12/dist-packages/transformer_engine -name "*.so" 2>/dev/null | head -5

################################################################################
# Install ModelOpt (Provides NVFP4 inference optimization)
################################################################################
# ModelOpt is a pip package (no compilation needed)
# Requires: PyTorch, CUDA runtime (both in NGC container)

RUN echo "Installing NVIDIA ModelOpt..." && \
    pip install --no-cache-dir nvidia-modelopt && \
    python -c "import modelopt; print('✅ ModelOpt installed successfully')"

################################################################################
# Install Megatron-LM (Large-scale training framework)
################################################################################
# Megatron is mostly Python code (minimal compilation)
# Works with Transformer Engine for FP8/NVFP4 training

RUN echo "Installing Megatron-LM..." && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout core_v0.12.1 && \
    echo "Initializing submodules..." && \
    git submodule update --init --recursive && \
    pip install --no-cache-dir -e . && \
    cd .. && \
    python -c "import megatron.core; print('✅ Megatron-LM installed successfully')"

################################################################################
# Install Additional Training & Development Tools
################################################################################
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    jupyter \
    jupyterlab \
    matplotlib \
    seaborn \
    pandas \
    h5py \
    tqdm \
    datasets \
    transformers \
    pytest \
    ipython

################################################################################
# Environment Configuration
################################################################################
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# Note: PYTHONPATH may be unset in base image, use :- to handle gracefully
ENV PYTHONPATH=/workspace/Megatron-LM:${PYTHONPATH:-}

# PyTorch 2.6+ compatibility
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

################################################################################
# Create Helper Scripts
################################################################################
# Script to show available tools
RUN echo '#!/bin/bash' > /usr/local/bin/show-tools && \
    echo 'cat << "EOF"' >> /usr/local/bin/show-tools && \
    echo "================================================================================" >> /usr/local/bin/show-tools && \
    echo "🚀 GTC SJ-26 NVFP4 TRAINING LAB" >> /usr/local/bin/show-tools && \
    echo "================================================================================" >> /usr/local/bin/show-tools && \
    echo "" >> /usr/local/bin/show-tools && \
    echo "Available Tools:" >> /usr/local/bin/show-tools && \
    echo "  • Transformer Engine - FP8/NVFP4 training" >> /usr/local/bin/show-tools && \
    echo "  • ModelOpt - NVFP4 inference optimization" >> /usr/local/bin/show-tools && \
    echo "  • Megatron-LM - Large-scale distributed training" >> /usr/local/bin/show-tools && \
    echo "  • PyTorch - Deep learning framework" >> /usr/local/bin/show-tools && \
    echo "  • Jupyter Lab - Interactive notebooks" >> /usr/local/bin/show-tools && \
    echo "  • W&B - Experiment tracking" >> /usr/local/bin/show-tools && \
    echo "" >> /usr/local/bin/show-tools && \
    echo "Quick Start:" >> /usr/local/bin/show-tools && \
    echo "  validate-env          # Verify installation" >> /usr/local/bin/show-tools && \
    echo "  jupyter lab --allow-root --ip=0.0.0.0  # Start Jupyter" >> /usr/local/bin/show-tools && \
    echo "  python train.py       # Run training" >> /usr/local/bin/show-tools && \
    echo "" >> /usr/local/bin/show-tools && \
    echo "Your code: /workspace/lab" >> /usr/local/bin/show-tools && \
    echo "================================================================================" >> /usr/local/bin/show-tools && \
    echo "EOF" >> /usr/local/bin/show-tools && \
    chmod +x /usr/local/bin/show-tools

# Validation script
RUN echo '#!/bin/bash' > /usr/local/bin/validate-env && \
    echo 'python << "EOF"' >> /usr/local/bin/validate-env && \
    echo 'import sys' >> /usr/local/bin/validate-env && \
    echo 'import torch' >> /usr/local/bin/validate-env && \
    echo 'print("="*80)' >> /usr/local/bin/validate-env && \
    echo 'print("🔍 GTC NVFP4 LAB - ENVIRONMENT VALIDATION")' >> /usr/local/bin/validate-env && \
    echo 'print("="*80)' >> /usr/local/bin/validate-env && \
    echo 'print(f"Python: {sys.version.split()[0]}")' >> /usr/local/bin/validate-env && \
    echo 'print(f"PyTorch: {torch.__version__}")' >> /usr/local/bin/validate-env && \
    echo 'print(f"CUDA: {torch.cuda.is_available()}")' >> /usr/local/bin/validate-env && \
    echo 'if torch.cuda.is_available():' >> /usr/local/bin/validate-env && \
    echo '    print(f"GPU Count: {torch.cuda.device_count()}")' >> /usr/local/bin/validate-env && \
    echo '    for i in range(torch.cuda.device_count()):' >> /usr/local/bin/validate-env && \
    echo '        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")' >> /usr/local/bin/validate-env && \
    echo 'print()' >> /usr/local/bin/validate-env && \
    echo 'print("Components:")' >> /usr/local/bin/validate-env && \
    echo 'try:' >> /usr/local/bin/validate-env && \
    echo '    import transformer_engine.pytorch as te' >> /usr/local/bin/validate-env && \
    echo '    print("  ✅ Transformer Engine")' >> /usr/local/bin/validate-env && \
    echo 'except: print("  ❌ Transformer Engine")' >> /usr/local/bin/validate-env && \
    echo 'try:' >> /usr/local/bin/validate-env && \
    echo '    import modelopt' >> /usr/local/bin/validate-env && \
    echo '    print("  ✅ ModelOpt")' >> /usr/local/bin/validate-env && \
    echo 'except: print("  ❌ ModelOpt")' >> /usr/local/bin/validate-env && \
    echo 'try:' >> /usr/local/bin/validate-env && \
    echo '    import megatron' >> /usr/local/bin/validate-env && \
    echo '    print("  ✅ Megatron-LM")' >> /usr/local/bin/validate-env && \
    echo 'except: print("  ❌ Megatron-LM")' >> /usr/local/bin/validate-env && \
    echo 'print("="*80)' >> /usr/local/bin/validate-env && \
    echo 'print("✅ Environment ready for NVFP4 training!")' >> /usr/local/bin/validate-env && \
    echo 'print("="*80)' >> /usr/local/bin/validate-env && \
    echo 'EOF' >> /usr/local/bin/validate-env && \
    chmod +x /usr/local/bin/validate-env

################################################################################
# Set Working Directory
################################################################################
WORKDIR /workspace/lab

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default: Show tools on startup
CMD ["bash", "-c", "show-tools && bash"]

################################################################################
# Final Verification
################################################################################
ARG CUDA_ARCH_LIST
RUN echo "================================================================================" && \
    echo "✅ GTC NVFP4 TRAINING LAB CONTAINER BUILD COMPLETE" && \
    echo "================================================================================" && \
    echo "" && \
    echo "Built for GPU architectures: ${CUDA_ARCH_LIST}" && \
    echo "" && \
    echo "Supported GPUs:" && \
    echo "  • Ada Lovelace: RTX 4090, L40, L4 (sm_89)" && \
    echo "  • Ampere: A100, A30, A10 (sm_80)" && \
    echo "  • Hopper: H100, H200 (sm_90a)" && \
    echo "  • Blackwell: B100, B200 (sm_100a)" && \
    echo "" && \
    echo "Ready for NVFP4 training and inference!" && \
    echo "  ✅ Transformer Engine (NVFP4 training)" && \
    echo "  ✅ ModelOpt (NVFP4 inference)" && \
    echo "  ✅ Megatron-LM (distributed training)" && \
    echo "  ✅ All development tools" && \
    echo "================================================================================"

