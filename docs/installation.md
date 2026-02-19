# Installation
Our code is tested on the following environment.

## 1. Create conda environment
```bash
conda create -n selfocc python=3.8.16
conda activate selfocc
```

## 2. Install PyTorch
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install packages from MMLab
```bash
pip install openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmsegmentation==1.0.0
mim install mmdet3d==1.1.1
```

## 4. Install other packages
```bash
pip install spconv-cu117
pip install timm
pip install wandb
```

## 4. Install custom CUDA ops
```bash
cd model/encoder/gaussian_encoder/ops && pip install -e .
cd model/head/localagg && pip install -e .
# for GaussianFormer-2
cd model/head/localagg_prob && pip install -e .
cd model/head/localagg_prob_fast && pip install -e .
```

## 5. (Optional) For visualization
```bash
pip install pyvirtualdisplay mayavi matplotlib==3.7.2 PyQt5
pip install rerun-sdk
pip install open3d
```




# 1. 환경 변수 설정 (Blackwell 최적화)
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1

# 2. 직접 설치 (pip 대신 setup.py 사용 추천)
python setup.py develop


# Local Aggregation Ops
cd model/encoder/gaussian_encoder/ops 
python setup.py develop

cd /workspace/GaussianFormer/model/head/localagg
python setup.py develop

cd ../localagg_prob
python setup.py develop

cd ../localagg_prob_fast
python setup.py develop


# Docker container setup(Blackwell CUDA 12.8)
