#!/bin/bash
################################################################################
# Save Docker Image to File for Distribution
#
# Purpose: Export the built container to a tar.gz file
# Use case: Distribute to students/developers who can't access registries
#
# Usage: ./docker/save-image.sh [output-path]
#
# Example:
#   ./docker/save-image.sh                    # Saves to ./gtc-nvfp4-lab.tar.gz
#   ./docker/save-image.sh /shared/images/    # Saves to custom location
#
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

IMAGE_NAME="gtc-nvfp4-lab"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Check if image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${FULL_IMAGE}$"; then
    echo "Image not found: ${FULL_IMAGE}"
    echo "Build it first: ./docker/build.sh"
    exit 1
fi

# Determine output path
if [ -z "$1" ]; then
    OUTPUT_DIR="."
else
    OUTPUT_DIR="$1"
    mkdir -p "$OUTPUT_DIR"
fi

OUTPUT_FILE="${OUTPUT_DIR}/gtc-nvfp4-lab.tar.gz"

echo -e "${BLUE}Saving Docker image to file...${NC}"
echo "Image: ${FULL_IMAGE}"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Get image size
IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${IMAGE_TAG} --format '{{.Size}}')
echo "Image size: ${IMAGE_SIZE}"
echo "Estimated file size: ~${IMAGE_SIZE} (compressed)"
echo ""
echo "This may take 3-5 minutes..."
echo ""

# Save and compress
docker save ${FULL_IMAGE} | gzip > ${OUTPUT_FILE}

if [ $? -eq 0 ]; then
    SAVED_SIZE=$(du -h ${OUTPUT_FILE} | cut -f1)
    echo ""
    echo -e "${GREEN}✅ Image saved successfully${NC}"
    echo ""
    echo "File: ${OUTPUT_FILE}"
    echo "Size: ${SAVED_SIZE}"
    echo ""
    echo "To distribute to students/developers:"
    echo "  1. Share file: ${OUTPUT_FILE}"
    echo "  2. Load with: docker load < gtc-nvfp4-lab.tar.gz"
    echo "  3. Run with: ./docker/run.sh"
else
    echo "❌ Save failed"
    exit 1
fi

