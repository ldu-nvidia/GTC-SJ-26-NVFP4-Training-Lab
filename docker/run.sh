#!/bin/bash
################################################################################
# Run GTC NVFP4 Training Lab Container
#
# Usage:
#   ./docker/run.sh                    # Interactive shell
#   ./docker/run.sh --validate         # Verify installation
#   ./docker/run.sh jupyter            # Start Jupyter Lab
#   ./docker/run.sh python script.py   # Run Python script
#
################################################################################

set -e

IMAGE_NAME="gtc-nvfp4-lab"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${FULL_IMAGE}$"; then
    echo "Image not found: ${FULL_IMAGE}"
    echo "Build it first: ./docker/build.sh"
    exit 1
fi

# Mount points
MOUNT_OPTS="-v ${PROJECT_ROOT}:/workspace/lab"

# Create directories if they don't exist
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/checkpoints"
mkdir -p "${PROJECT_ROOT}/logs"

# Mount additional directories
MOUNT_OPTS="$MOUNT_OPTS -v ${PROJECT_ROOT}/data:/workspace/data"
MOUNT_OPTS="$MOUNT_OPTS -v ${PROJECT_ROOT}/checkpoints:/workspace/checkpoints"
MOUNT_OPTS="$MOUNT_OPTS -v ${PROJECT_ROOT}/logs:/workspace/logs"

# Environment variables
ENV_OPTS=""
if [ ! -z "$WANDB_API_KEY" ]; then
    ENV_OPTS="$ENV_OPTS -e WANDB_API_KEY=$WANDB_API_KEY"
fi

# Validation function
validate_env() {
    echo "Validating environment..."
    docker run --gpus all --rm \
        --ipc=host \
        $MOUNT_OPTS \
        $ENV_OPTS \
        $FULL_IMAGE \
        validate-env
}

# Jupyter function
run_jupyter() {
    echo "Starting Jupyter Lab..."
    echo "Access at: http://localhost:8888"
    echo "Press Ctrl+C to stop"
    docker run --gpus all --rm -it \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p 8888:8888 \
        -p 6006:6006 \
        $MOUNT_OPTS \
        $ENV_OPTS \
        --workdir /workspace/lab \
        $FULL_IMAGE \
        jupyter lab --allow-root --ip=0.0.0.0 --no-browser
}

# Handle special commands
if [ "$1" == "--validate" ] || [ "$1" == "-v" ]; then
    validate_env
    exit 0
elif [ "$1" == "jupyter" ]; then
    run_jupyter
    exit 0
fi

# Regular command execution
if [ $# -eq 0 ]; then
    echo "Starting GTC NVFP4 Lab container..."
    echo "Tip: Run 'show-tools' inside container for quick start guide"
    echo ""
    docker run --gpus all --rm -it \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p 8888:8888 \
        -p 6006:6006 \
        $MOUNT_OPTS \
        $ENV_OPTS \
        --workdir /workspace/lab \
        $FULL_IMAGE \
        /bin/bash
else
    docker run --gpus all --rm -it \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        $MOUNT_OPTS \
        $ENV_OPTS \
        --workdir /workspace/lab \
        $FULL_IMAGE \
        "$@"
fi

