## 1. Container run 
```bash
docker run --gpus '"device=6,7"' -it \
    -v /disks/ssd1/kmw2622/project/3d_occupancy_prediction/GaussianFormer:/workspace/GaussianFormer \
    -v /disks/ssd1/kmw2622/dataset/nuscenes:/workspace/GaussianFormer/data/nuscenes \
    --ipc=host \
    gaussian_former_v1:latest /bin/bash
```