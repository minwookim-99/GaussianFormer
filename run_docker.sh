#!/bin/bash

# GPU 
GPUS='"device=6,7"'
CONTAINER_NAME="gaussian_former_v1"

# Dataset directory 
PROJECT_PATH="/disks/ssd1/kmw2622/project/3d_occupancy_prediction/GaussianFormer"
NUSCENE_DATASET_PATH="/disks/ssd1/kmw2622/dataset/nuscenes"
NUSCENE_CAM_DATASET_PATH="/disks/ssd1/kmw2622/dataset/nuscenes_cam"
SURROUNDOCC_DATASET_PATH="/disks/ssd1/kmw2622/dataset/surroundocc"

echo "🚀 Starting GaussianFormer Docker Environment on GPUs $GPUS..."

docker run --gpus $GPUS -it \
    --name $CONTAINER_NAME \
    -v $PROJECT_PATH:/workspace/GaussianFormer \
    -v $NUSCENE_DATASET_PATH:/workspace/GaussianFormer/data/nuscenes \
    -v $NUSCENE_CAM_DATASET_PATH:/workspace/GaussianFormer/data/nuscenes_cam \
    -v $SURROUNDOCC_DATASET_PATH:/workspace/GaussianFormer/data/surroundocc \
    --ipc=host \
    gaussian_former_v1:latest /bin/bash