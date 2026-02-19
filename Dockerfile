# 1. 베이스 이미지 설정 (PyTorch 2.8 / CUDA 12.8 - Blackwell 대응)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# 2. Blackwell(12.0) 아키텍처 및 빌드 환경 변수 설정
# RTX 6000 Blackwell은 Compute Capability 12.0을 사용합니다.
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV FORCE_CUDA=1
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8 

# 3. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    ninja-build wget build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. 빌드 도구 및 파이썬 환경 준비
# pkg_resources 에러 방지를 위해 setuptools 버전을 69.5.1로 고정합니다.
RUN pip install --upgrade pip
RUN pip install "setuptools==69.5.1" "wheel<1.0.0"

WORKDIR /workspace

# --- MM-시리즈 소스 빌드 단계 (버전 체크 패치 포함) ---

# 5. MMCV 빌드 (v2.1.0)
RUN git clone https://github.com/minwookim-99/mmcv.git && cd mmcv && \
    git checkout v2.1.0 && \
    pip install -r requirements.txt && \
    python setup.py build_ext --inplace && \
    pip install -e . --no-build-isolation

# 6. MMDetection 빌드 (v3.0.0) 및 패치
RUN git clone https://github.com/minwookim-99/mmdetection.git && cd mmdetection && \
    git checkout v3.0.0-cuda12.8-blackwell && \
    pip install -v -e . --no-build-isolation

# 7. MMSegmentation 빌드 (v1.0.0) 및 패치
RUN git clone https://github.com/minwookim-99/mmsegmentation.git && cd mmsegmentation && \
    git checkout v1.0.0-cuda12.8-blackwell && \
    pip install -v -e . --no-build-isolation

# 8. MMDetection3D 빌드 (v1.1.1) 및 패치
RUN git clone https://github.com/minwookim-99/mmdetection3d.git && cd mmdetection3d && \
    git checkout v1.1.1-cuda12.8-blackwell && \
    pip install -v -e . --no-build-isolation

# --- 프로젝트 전용 설정 및 커스텀 Ops ---

# 9. 기타 필수 패키지 (Blackwell 호환 spconv)
RUN pip install spconv-cu120 timm pyvirtualdisplay einops jaxtyping && \
    pip install "matplotlib<3.6.0" PyQt5

# 10. GaussianFormer 프로젝트 복사 및 커스텀 Ops 빌드
# 주의: 소스 코드 폴더명이 GaussianFormer라고 가정합니다.
WORKDIR /workspace/GaussianFormer
COPY . .

# (1) Gaussian Encoder Ops
RUN cd model/encoder/gaussian_encoder/ops && \
    rm -rf build *.so *.egg-info && \
    python setup.py build_ext --inplace && \
    # 패키지 구조에 따라 필요시 .so 파일 위치 보정 
    pip install -e . --no-build-isolation

# (2) Local Aggregation Ops 
RUN cd model/head/localagg && \
    rm -rf build *.so *.egg-info && \
    python setup.py build_ext --inplace && \
    # 이 모듈은 local_aggregate/__init__.py에서 상대 임포트를 하므로 복사가 필수입니다.
    #cp _C*.so local_aggregate/ && \
    pip install -e . --no-build-isolation

# (3) Local Aggregation Prob (GaussianFormer-2용)
RUN cd model/head/localagg_prob && \
    rm -rf build *.so *.egg-info && \
    python setup.py build_ext --inplace && \
    #cp _C*.so local_aggregate_prob/ 2>/dev/null || true && \
    pip install -e . --no-build-isolation

# (4) Local Aggregation Prob Fast (GaussianFormer-2용)
RUN cd model/head/localagg_prob_fast && \
    rm -rf build *.so *.egg-info && \
    python setup.py build_ext --inplace && \
    #cp _C*.so local_aggregate_prob_fast/ 2>/dev/null || true && \
    pip install -e . --no-build-isolation
    
WORKDIR /workspace/GaussianFormer