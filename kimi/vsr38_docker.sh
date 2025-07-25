#!/bin/bash

# Path to your project on the Jetson Orin
# HOST_PROJECT_DIR="/home/dj/projects/my_model" -v "${HOST_PROJECT_DIR}":/workspace/my_model \

# Name of the NVIDIA PyTorch Docker image (JetPack 5.0.2 compatible)
# IMAGE_NAME="nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3"
IMAGE_NAME="vsr38_docker:latest"

# Run Docker container
sudo docker run -it --rm --runtime nvidia \
  --network host \
  --volume ~:/workspace \
  -w /workspace/my_model \
  "${IMAGE_NAME}"
