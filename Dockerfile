# 1. 베이스 이미지 선택 (PyTorch와 CUDA가 깔린 이미지 사용)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 시스템 패키지 및 Python 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-distutils git wget \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 기본 명령어 연결 (선택사항이지만 편리함)
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# 3. uv 설치 (가장 빠른 방법)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 5. uv를 이용한 의존성 설치 (캐시 활용을 위해 requirements부터)
# SelfOcc는 pip install 명령어가 많으므로 하나씩 실행하거나 script화 합니다.
RUN uv pip install --system \
    torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

RUN apt-get purge -y python3-blinker || true

# 6. MMLab 패키지 설치 (MIM 사용)
RUN uv pip install --system --break-system-packages openmim && \
    mim install mmcv==2.0.1 mmdet==3.0.0 mmsegmentation==1.0.0 mmdet3d==1.1.1

# 7. 기타 패키지
RUN uv pip install --system spconv-cu117 timm

# 작업 디렉토리 설정
WORKDIR /workspace/GaussianFormer

# 8. 소스 코드 복사 및 Custom Ops 빌드
COPY . .

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"
ENV FORCE_CUDA="1"

RUN cd model/encoder/gaussian_encoder/ops && pip install -e .
RUN cd model/head/localagg && pip install -e .
RUN cd model/head/localagg_prob && pip install -e .
RUN cd model/head/localagg_prob_fast && pip install -e .
# ... 나머지 빌드 과정 진행