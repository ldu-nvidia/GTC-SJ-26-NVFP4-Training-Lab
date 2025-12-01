#!/bin/bash
################################################################################
# Load Docker Image from File
#
# Purpose: Load a saved Docker image (for students/developers)
# Usage: ./docker/load-image.sh [image-file]
#
################################################################################

set -e

IMAGE_FILE="${1:-gtc-nvfp4-lab.tar.gz}"

if [ ! -f "$IMAGE_FILE" ]; then
    echo "Image file not found: $IMAGE_FILE"
    echo "Usage: ./docker/load-image.sh [image-file.tar.gz]"
    exit 1
fi

echo "Loading Docker image from: $IMAGE_FILE"
echo "This may take 2-3 minutes..."
echo ""

docker load < "$IMAGE_FILE"

echo ""
echo "✅ Image loaded successfully"
echo ""
echo "Run with: ./docker/run.sh"

