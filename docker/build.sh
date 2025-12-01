#!/bin/bash
################################################################################
# Build GTC NVFP4 Training Lab Container
#
# This builds a complete environment with:
#   - Transformer Engine (NVFP4 training)
#   - ModelOpt (NVFP4 inference)
#   - Megatron-LM (large-scale training)
#   - All development tools
#
# Usage: ./docker/build.sh
# Time: ~15-20 minutes
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_NAME="gtc-nvfp4-lab"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🚀 GTC SJ-26 NVFP4 TRAINING LAB - BUILD CONTAINER${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Auto-detect GPU architecture or use default
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}Detecting GPU architecture...${NC}"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "Detected GPU: $GPU_NAME"
    
    # Map GPU names to CUDA architectures
    if [[ "$GPU_NAME" == *"RTX 40"* ]] || [[ "$GPU_NAME" == *"RTX 4090"* ]] || [[ "$GPU_NAME" == *"L40"* ]] || [[ "$GPU_NAME" == *"L4"* ]]; then
        CUDA_ARCH="8.9"
        echo "→ Architecture: sm_89 (Ada Lovelace)"
    elif [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"A30"* ]] || [[ "$GPU_NAME" == *"A10"* ]]; then
        CUDA_ARCH="8.0"
        echo "→ Architecture: sm_80 (Ampere)"
    elif [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"H200"* ]]; then
        CUDA_ARCH="9.0"
        echo "→ Architecture: sm_90a (Hopper)"
    elif [[ "$GPU_NAME" == *"Blackwell"* ]] || [[ "$GPU_NAME" == *"B100"* ]] || [[ "$GPU_NAME" == *"B200"* ]]; then
        # Blackwell can be sm_100a or sm_103a depending on variant
        CUDA_ARCH="10.0"
        echo "→ Architecture: sm_100a/sm_103a (Blackwell)"
    elif [[ "$GPU_NAME" == *"V100"* ]]; then
        CUDA_ARCH="7.0"
        echo "→ Architecture: sm_70 (Volta)"
    else
        # Default: build for common architectures (Ada, Ampere, Hopper, Blackwell)
        CUDA_ARCH="8.0;8.9;9.0;10.0"
        echo -e "${YELLOW}Unknown GPU, building for Ada/Ampere/Hopper/Blackwell${NC}"
    fi
else
    # No GPU detected, use comprehensive default
    CUDA_ARCH="8.0;8.9;9.0;10.0"
    echo -e "${YELLOW}No GPU detected, building for Ada/Ampere/Hopper/Blackwell${NC}"
fi

# Allow manual override
if [ ! -z "$CUDA_ARCH_LIST" ]; then
    CUDA_ARCH="$CUDA_ARCH_LIST"
    echo -e "${BLUE}Manual override: CUDA_ARCH_LIST=${CUDA_ARCH}${NC}"
fi

echo ""
echo -e "${GREEN}Building for CUDA architectures: ${CUDA_ARCH}${NC}"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    echo "Install: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "✅ Docker: $(docker --version | cut -d' ' -f3 | cut -d',' -f1)"

# Check GPU access
if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo -e "✅ GPU accessible"
else
    echo -e "${YELLOW}⚠️  GPU not accessible (install nvidia-docker2)${NC}"
fi

# Check disk space
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE" -lt 20 ]; then
    echo -e "${RED}❌ Need 20GB, have ${AVAILABLE}GB${NC}"
    exit 1
fi
echo -e "✅ Disk space: ${AVAILABLE}GB"

echo ""
echo "Components to build:"
echo "  1. NVIDIA PyTorch NGC base (CUDA 12.6+)"
echo "  2. Transformer Engine (NVFP4 training)"
echo "  3. ModelOpt (NVFP4 inference optimization)"
echo "  4. Megatron-LM (distributed training)"
echo "  5. Development tools (Jupyter, W&B, etc.)"
echo ""
echo -e "${BLUE}Starting build (~15-20 minutes)...${NC}"
echo ""

docker build \
  --progress=plain \
  --build-arg CUDA_ARCH_LIST="$CUDA_ARCH" \
  -t "$FULL_IMAGE" \
  -f docker/Dockerfile \
  .

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}✅ BUILD COMPLETE: ${FULL_IMAGE}${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "Size: $(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format '{{.Size}}')"
echo ""
echo "📦 What's included:"
echo "  ✅ CUDA 12.6+ & PyTorch 2.8+"
echo "  ✅ Transformer Engine (NVFP4 training)"
echo "  ✅ ModelOpt (NVFP4 inference)"
echo "  ✅ Megatron-LM (distributed training)"
echo "  ✅ Jupyter Lab"
echo "  ✅ W&B, TensorBoard"
echo ""
echo -e "${BLUE}Quick start:${NC}"
echo "  ./docker/run.sh                # Interactive shell"
echo "  ./docker/run.sh --validate     # Verify installation"
echo "  ./docker/run.sh jupyter        # Start Jupyter Lab"
echo ""
echo -e "${GREEN}Ready for NVFP4 training! 🎉${NC}"

