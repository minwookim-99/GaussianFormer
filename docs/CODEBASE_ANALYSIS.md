# GaussianFormer 코드베이스 분석

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉터리 구조](#2-디렉터리-구조)
3. [모델 아키텍처](#3-모델-아키텍처)
4. [주요 모듈 상세](#4-주요-모듈-상세)
5. [설정 시스템](#5-설정-시스템)
6. [데이터 파이프라인](#6-데이터-파이프라인)
7. [손실 함수](#7-손실-함수)
8. [학습 및 평가 파이프라인](#8-학습-및-평가-파이프라인)
9. [핵심 알고리즘 및 기법](#9-핵심-알고리즘-및-기법)
10. [성능 벤치마크](#10-성능-벤치마크)
11. [환경 설정](#11-환경-설정)
12. [실행 방법](#12-실행-방법)

---

## 1. 프로젝트 개요

**GaussianFormer**는 자율주행 차량의 다중 카메라 이미지로부터 3D 시맨틱 점유 예측(3D Semantic Occupancy Prediction)을 수행하는 모델이다. 두 가지 버전을 포함한다.

| 버전 | 논문 | 핵심 기여 |
|------|------|-----------|
| **GaussianFormer v1** | ECCV 2024 | Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction |
| **GaussianFormer-2** | CVPR 2025 | Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction |

### 핵심 아이디어

기존 밀집 복셀 그리드(200×200×16 = 640K 복셀) 방식 대신, **3D 가우시안 분포**를 객체 중심 표현으로 활용한다.

- 약 25K 개의 가우시안 = 96% 메모리 절감
- 연속적인 표현으로 불확실성/신뢰도 자연스럽게 인코딩
- CUDA 기반 가우시안→복셀 스플래팅으로 고속 렌더링

---

## 2. 디렉터리 구조

```
GaussianFormer/
├── train.py                    # 분산 학습 스크립트 (DDP)
├── eval.py                     # 평가 스크립트
├── visualize.py                # 예측 및 가우시안 시각화
├── vis.py                      # 추가 시각화 유틸리티
├── Dockerfile                  # CUDA 12.8 환경 Docker 설정
│
├── config/                     # MMEngine 설정 파일
│   ├── _base_/
│   │   ├── model.py            # GaussianFormer v1 기본 아키텍처
│   │   ├── misc.py             # 학습 하이퍼파라미터
│   │   └── surroundocc.py      # 데이터 파이프라인
│   ├── nuscenes_gs144000.py    # v1 기본 설정 (144K 가우시안)
│   ├── nuscenes_gs25600_solid.py  # v1 비어있지 않은 복셀 변형
│   └── prob/                   # GaussianFormer-2 설정
│       ├── nuscenes_gs6400.py  # 경량 버전
│       ├── nuscenes_gs12800.py # 중간 버전
│       └── nuscenes_gs25600.py # 전체 버전 (SOTA)
│
├── model/                      # 모델 구현
│   ├── backbone/               # 이미지 특징 추출 (ResNet 등)
│   ├── neck/                   # 특징 피라미드 네트워크 (FPN)
│   ├── lifter/                 # 2D→3D 리프팅 (가우시안 초기화)
│   │   ├── gaussian_lifter.py           # v1 리프터
│   │   ├── gaussian_lifter_v2.py        # v2 확률적 리프터
│   │   └── gaussian_initializer/        # ResNet 기반 점유 분포 예측
│   ├── encoder/                # 가우시안 인코더
│   │   └── gaussian_encoder/
│   │       ├── gaussian_encoder.py      # 핵심 인코더 루프
│   │       ├── anchor_encoder_module.py # 가우시안 파라미터 임베딩
│   │       ├── deformable_module.py     # 변형 가능 어텐션 + 특징 집계
│   │       ├── refine_module.py         # v1 정제 모듈
│   │       ├── refine_module_v2.py      # v2 확률적 정제 모듈
│   │       ├── spconv3d_module.py       # 희소 3D 합성곱
│   │       ├── ffn_module.py            # 피드포워드 네트워크
│   │       └── ops/                     # 커스텀 CUDA 연산
│   ├── head/                   # 점유 예측 헤드
│   │   ├── gaussian_head.py             # 메인 헤드 (로컬 집계 사용)
│   │   ├── localagg/                    # CUDA: 가우시안→복셀 스플래팅
│   │   ├── localagg_prob/               # 확률적 집계 (v2)
│   │   ├── localagg_prob_fast/          # 빠른 확률적 집계
│   │   └── base_head.py
│   ├── segmentor/
│   │   ├── base_segmentor.py
│   │   └── bev_segmentor.py    # 전체 파이프라인 통합 모델
│   └── utils/
│       ├── safe_ops.py         # 수치 안정 연산
│       ├── sampler.py
│       └── utils.py            # 회전 행렬, 유틸리티
│
├── dataset/
│   ├── dataset.py              # NuScenes 데이터셋 로더
│   ├── sampler.py
│   ├── transform_3d.py         # 데이터 증강
│   └── utils.py
│
├── loss/
│   ├── occupancy_loss.py       # CE + Lovász + Dice 복합 손실
│   ├── bce_loss.py
│   ├── multi_loss.py           # 다중 손실 결합
│   ├── base_loss.py
│   └── utils/
│       └── lovasz_softmax.py
│
├── misc/
│   ├── metric_util.py          # mIoU 메트릭
│   ├── checkpoint_util.py      # 체크포인트 관리
│   └── tb_wrapper.py           # TensorBoard 로깅
│
└── docs/
    ├── installation.md
    └── run.md
```

---

## 3. 모델 아키텍처

### 전체 데이터 흐름

```
다중 뷰 이미지 (B, 6, 3, H, W)
         │
         ▼
┌─────────────────────┐
│  Image Backbone     │  ResNet50/101 - 4단계 다중 스케일 특징
│  + Image Neck (FPN) │  FPN으로 특징 융합
└─────────────────────┘
         │  (B, 6, C, H', W') × 4 레벨
         ▼
┌─────────────────────┐
│  Lifter             │  3D 가우시안 앵커 초기화
│  GaussianLifter(V2) │  representation (B, G, 11)
└─────────────────────┘  rep_features (B, G, C)
         │
         ▼
┌─────────────────────┐
│  Encoder            │  6개 디코더 레이어 반복:
│  GaussianOccEncoder │  ① SpConv3D  → 가우시안 간 공간 관계
│                     │  ② LayerNorm
│                     │  ③ DeformableAttention → 이미지 특징 융합
│                     │  ④ FFN       → 가우시안별 특징 변환
│                     │  ⑤ Refine    → 가우시안 파라미터 업데이트
└─────────────────────┘
         │  정제된 가우시안 파라미터
         ▼
┌─────────────────────┐
│  Head               │  CUDA 로컬 집계 (가우시안→복셀 스플래팅)
│  GaussianHead       │  점유 그리드 생성 및 시맨틱 레이블 할당
└─────────────────────┘
         │
         ▼
점유 예측: (B, 200, 200, 16, 18 클래스)
```

### 가우시안 파라미터

각 가우시안은 다음 파라미터로 정의된다.

| 파라미터 | 차원 | 설명 |
|---------|------|------|
| **Position (xyz)** | 3 | 3D 중심 좌표 |
| **Scale** | 3 | 대각선 공분산 스케일링 |
| **Rotation** | 4 | 정규화된 쿼터니언 |
| **Opacity** | 1 | 불투명도/신뢰도 |
| **Semantics** | 17 | 시맨틱 클래스 로짓 |

---

## 4. 주요 모듈 상세

### 4.1 BEVSegmentor (`model/segmentor/bev_segmentor.py`)

전체 파이프라인을 통합하는 메인 모델 클래스.

```python
class BEVSegmentor(BaseSegmentor):
    def forward(self, imgs, metas):
        # 1. 이미지 백본 + 넥으로 다중 스케일 특징 추출
        img_feats = self.extract_img_feat(imgs)  # List[Tensor]

        # 2. 2D→3D 리프팅: 가우시안 초기화
        representation, instance_feature = self.lifter(img_feats, metas)

        # 3. 가우시안 인코딩 (반복 정제)
        representation, instance_feature = self.encoder(
            representation, instance_feature, img_feats, metas
        )

        # 4. 점유 헤드: 가우시안→복셀 변환
        output = self.head(representation, instance_feature, metas)

        return output
```

### 4.2 GaussianLifter (`model/lifter/gaussian_lifter.py`)

학습 가능한 가우시안 앵커를 초기화하고 특징을 임베딩한다.

**핵심 특징**:
- `num_anchor`개의 학습 가능한 앵커 파라미터 초기화
- 인스턴스 특징 벡터 생성 (각 가우시안마다 `embed_dims` 차원)
- 좌표계: 극좌표(polar) 또는 데카르트(cartesian)

### 4.3 GaussianLifterV2 (`model/lifter/gaussian_lifter_v2.py`)

GaussianFormer-2의 분포 기반 초기화.

**v1과의 차이점**:
- ResNet 기반 초기화기로 픽셀별 점유 분포 예측
- 예측된 분포에서 깊이 샘플링하여 가우시안 위치 결정
- 더 안정적이고 정보가 풍부한 초기화

### 4.4 GaussianOccEncoder (`model/encoder/gaussian_encoder/gaussian_encoder.py`)

가우시안을 반복적으로 정제하는 핵심 인코더.

**레이어 순서** (설정 가능):
```python
# 기본 순서: ['spconv', 'norm', 'deformable', 'ffn', 'refine']
operation_order = ['norm', 'deformable', 'norm', 'ffn',
                   'norm', 'refine', 'spconv']
```

**각 연산의 역할**:
- `spconv`: 희소 3D 합성곱으로 가우시안 간 공간 관계 모델링
- `deformable`: 변형 가능 어텐션으로 이미지 특징 융합
- `ffn`: 피드포워드 네트워크로 가우시안별 특징 변환
- `refine`: 가우시안 파라미터(위치, 스케일, 회전 등) 업데이트

### 4.5 DeformableFeatureAggregation (`model/encoder/gaussian_encoder/deformable_module.py`)

이미지 특징을 가우시안으로 집계하는 핵심 어텐션 모듈.

**동작 방식**:
1. 각 가우시안 중심 주변에 샘플링 포인트 생성
   - 고정 오프셋 포인트 (중심 + 6방향)
   - 인스턴스 특징에서 예측된 학습 가능 포인트
2. 포인트를 카메라 좌표계로 투영
3. 4개 FPN 레벨에서 이미지 특징 바이리니어 샘플링
4. 그룹별 어텐션 가중치로 특징 집계

### 4.6 SparseGaussian3DRefinementModule (`model/encoder/gaussian_encoder/refine_module.py`)

가우시안 파라미터를 예측하고 업데이트하는 정제 모듈.

**예측 내용**:
- 위치 델타 (xyz 오프셋)
- 스케일 업데이트
- 회전 쿼터니언 업데이트
- 불투명도 업데이트
- 시맨틱 로짓

### 4.7 GaussianHead (`model/head/gaussian_head.py`)

가우시안을 3D 점유 그리드로 변환하는 헤드.

**동작 방식**:
1. 가우시안 파라미터 준비 (평균, 스케일, 회전, 불투명도, 시맨틱)
2. CUDA `local_aggregate` 연산으로 가우시안→복셀 스플래팅
3. 손실 계산용 GT 복셀 샘플링
4. 예측 점유 그리드 반환

**GaussianFormer-2**: `localaggprob`으로 확률적 집계 사용

---

## 5. 설정 시스템

MMEngine 레지스트리 패턴으로 모든 설정 관리.

### 5.1 기본 모델 설정 (`config/_base_/model.py`)

```python
embed_dims = 128        # 특징 차원
num_decoder = 6         # 정제 레이어 수
num_groups = 4          # 변형 어텐션 그룹 수
num_levels = 4          # 다중 스케일 레벨 수
include_opa = True      # 불투명도 포함 여부
semantics = True        # 시맨틱 예측 여부
semantic_dim = 17       # NuScenes 17개 클래스
xyz_coordinate = 'polar'     # 좌표계
phi_activation = 'loop'      # 극좌표 활성화 함수
```

### 5.2 데이터 설정 (`config/_base_/surroundocc.py`)

```python
data_root = "data/nuscenes/"
batch_size = 1
input_shape = (704, 256)   # 입력 이미지 해상도
data_aug_conf = {
    "resize_lim": (0.40, 0.47),  # 랜덤 리사이즈 범위
    "rot_lim": (-5.4, 5.4),       # 회전 각도 범위 (도)
    "rand_flip": True              # 수평 플립
}
pc_range = [-50, -50, -5, 50, 50, 3]  # 장면 범위 (미터)
voxel_size = 0.5                        # 복셀 해상도 (미터)
occ_size = [200, 200, 16]              # 점유 그리드 크기
```

### 5.3 학습 설정 (`config/_base_/misc.py`)

```python
max_epochs = 20
optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)
grad_max_norm = 35     # 그래디언트 클리핑
warmup_iters = 500
scheduler = "CosineLRScheduler"  # 또는 MultiStepLR
```

### 5.4 GaussianFormer-2 경량 설정 (`config/prob/nuscenes_gs6400.py`)

```python
# 리프터
lifter = GaussianLifterV2(
    num_anchor = 4000,
    num_samples = 128,       # 픽셀당 깊이 샘플 수
    random_sampling = False,
    deterministic = False
)

# 인코더 (희소 합성곱 포함)
encoder = GaussianOccEncoder(
    num_decoder = 6,
    embed_dims = 128,
    # 연산 순서에 spconv 포함
)

# 헤드 (확률적 집계)
head = GaussianHead(
    use_localaggprob = True,   # 확률적 집계
    combine_geosem = True,
    H=200, W=200, D=16,
    pc_range = [-50, -50, -5, 50, 50, 3]
)
```

---

## 6. 데이터 파이프라인

### 6.1 NuScenes 데이터셋 (`dataset/dataset.py`)

**입력**:
- 6개 카메라 이미지 (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right)
- 카메라 내외부 파라미터 (intrinsics, extrinsics)
- 점유 GT (SurroundOcc 형식 `.npy` 파일)

**전처리 및 증강** (`dataset/transform_3d.py`):
- 랜덤 리사이즈 (0.40×~0.47× 원본)
- 랜덤 수평 플립
- 랜덤 회전 (-5.4°~5.4°)
- 색상 지터링 (선택적)

### 6.2 데이터 디렉터리 구조

```
data/
├── nuscenes/
│   ├── maps/
│   ├── samples/         # 키프레임 센서 데이터
│   ├── sweeps/          # 중간 프레임 센서 데이터
│   ├── v1.0-trainval/
│   └── v1.0-test/
├── nuscenes_cam/
│   ├── nuscenes_infos_train_sweeps_occ.pkl  # 학습 메타데이터
│   └── nuscenes_infos_val_sweeps_occ.pkl    # 검증 메타데이터
└── surroundocc/
    └── samples/
        └── xxxxxxxx.pcd.bin.npy             # 점유 GT 레이블
```

---

## 7. 손실 함수

### OccupancyLoss (`loss/occupancy_loss.py`)

다중 컴포넌트 손실 함수:

| 손실 | 가중치 | 설명 |
|------|--------|------|
| **Cross-Entropy Loss** | 10.0 | 복셀별 다중 클래스 분류 |
| **Lovász Softmax Loss** | 1.0 | 클래스 불균형 및 경계 처리 |
| **Semantic Scaling Loss** | 1.0 | 공간적 특성으로 가중된 시맨틱 |
| **Geometric Scaling Loss** | 1.0 | 공간적 특성으로 가중된 기하학적 |
| **Dice Loss** | 선택적 | 클래스 균형 추가 정규화 |
| **Focal Loss** | 선택적 | 어려운 샘플에 집중 |

**총 손실**: `L = 10×CE + 1×Lovász + 1×SemanticScal + 1×GeoScal`

---

## 8. 학습 및 평가 파이프라인

### 8.1 학습 흐름 (`train.py`)

1. **초기화**
   - DDP (NCCL 백엔드) 분산 학습 설정
   - MMEngine Config로 모델 빌드
   - 학습/검증 데이터로더 생성

2. **에폭당 루프**
   ```python
   for data in train_loader:
       result_dict = model(imgs=input_imgs, metas=data)
       loss = criterion(result_dict, gt_labels)
       loss.backward()

       # 그래디언트 누적 지원
       if step % grad_accum == 0:
           clip_grad_norm_(model.parameters(), max_norm=35)
           optimizer.step()
           scheduler.step()
   ```

3. **평가 및 체크포인팅**
   - `eval_every_epochs`마다 mIoU 메트릭 계산
   - `epoch_{n}.pth` 저장 + `latest.pth` 심링크

### 8.2 평가 메트릭 (`misc/metric_util.py`)

- **mIoU (mean Intersection over Union)**: 17개 시맨틱 클래스의 평균 IoU
- `MeanIoU` 클래스로 구현, 분산 학습에서 집계 지원

---

## 9. 핵심 알고리즘 및 기법

### 9.1 가우시안 혼합 표현

**동기**: 640K 복셀 vs 25K 가우시안 → 96% 메모리 절감

**가우시안이 효과적인 이유**:
- 가우시안 혼합 모델(GMM) 이론으로 임의의 분포 근사 가능
- 연속적인 표현 (이산 복셀과 달리)
- CUDA 기반 스플래팅으로 빠른 복셀 그리드 변환
- 불투명도로 불확실성/신뢰도 자연스럽게 표현

### 9.2 확률적 가우시안 중첩 (GaussianFormer-2)

기존 이진 점유 대신 각 가우시안이 확률 분포를 표현:

```
P(복셀 점유) = Π P_i(점유 | 가우시안 i에 인접)
```

**장점**:
- 물리적으로 더 타당한 표현
- 중첩된 가우시안 처리 개선
- 시맨틱과 기하학적 정보 통합 효율화

### 9.3 변형 가능 어텐션 (Deformable Attention)

각 가우시안 주변의 학습 가능한 샘플링 포인트에서 이미지 특징 집계:

```
샘플링 포인트 = 고정 오프셋 + 예측된 오프셋 (인스턴스 특징 기반)
특징 = Σ attention_weight × image_feature(projected_point)
```

- 4개 FPN 레벨에서 멀티스케일 샘플링
- 그룹별 어텐션 (채널을 그룹으로 분할)
- 카메라별 임베딩으로 로버스트성 향상

### 9.4 희소 3D 합성곱

전체 복셀 그리드 대신 가우시안 위치에서만 합성곱:

- `SubMConv3d` (SubManifold): 출력 희소도 = 입력 희소도
- FLOPs를 크게 줄이면서 기하학적 추론 유지
- `spconv` 라이브러리 사용

### 9.5 분포 기반 초기화 (GaussianFormer-2)

점 깊이 예측 대신:
1. ResNet 기반 초기화기로 픽셀별 점유 분포 예측
2. 예측된 분포에서 깊이 샘플링
3. 샘플링된 위치에 가우시안 배치

→ 점 깊이 예측보다 안정적이고 정보가 풍부한 초기화

---

## 10. 성능 벤치마크

nuScenes 점유 벤치마크 결과:

| 모델 | 버전 | 가우시안 수 | mIoU | 설정 파일 |
|------|------|------------|------|----------|
| Baseline | GaussianFormer v1 | 144,000 | 19.10% | `nuscenes_gs144000.py` |
| NonEmpty | GaussianFormer v1 | 25,600 | 19.31% | `nuscenes_gs25600_solid.py` |
| **Prob-64** | GaussianFormer-2 | 6,400 | 20.04% | `prob/nuscenes_gs6400.py` |
| **Prob-128** | GaussianFormer-2 | 12,800 | 20.08% | `prob/nuscenes_gs12800.py` |
| **Prob-256** | GaussianFormer-2 | 25,600 | **20.33%** | `prob/nuscenes_gs25600.py` |

**핵심 결과**:
- GaussianFormer-2가 25.6K 가우시안으로 **20.33% mIoU** 달성
- 밀집 점유 방법 대비 **75~82% 메모리 절감**
- nuScenes 점유 벤치마크 SOTA

---

## 11. 환경 설정

### 의존성

```bash
# PyTorch (CUDA 11.8 기준)
torch==2.0.0, torchvision==0.15.1, torchaudio==2.0.1

# MMLab 생태계
mmcv==2.0.1
mmdet==3.0.0
mmsegmentation==1.0.0
mmdet3d==1.1.1

# 기타 주요 패키지
spconv-cu117   # 희소 합성곱
timm           # 비전 트랜스포머 백본
einops         # 텐서 재배열
pyquaternion   # 쿼터니언 연산
```

### 설치 순서

```bash
# 1. conda 환경 생성
conda create -n selfocc python=3.8.16
conda activate selfocc

# 2. PyTorch 설치
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. MMLab 패키지 설치
pip install openmim
mim install mmcv==2.0.1 mmdet==3.0.0 mmsegmentation==1.0.0 mmdet3d==1.1.1

# 4. 기타 패키지
pip install spconv-cu117 timm einops pyquaternion

# 5. 커스텀 CUDA 연산 빌드 (필수)
cd model/encoder/gaussian_encoder/ops && pip install -e .
cd model/head/localagg && pip install -e .
cd model/head/localagg_prob && pip install -e .
cd model/head/localagg_prob_fast && pip install -e .
```

---

## 12. 실행 방법

### 학습

```bash
# 단일 GPU
python train.py \
    --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gs6400/

# 다중 GPU (8개)
python -m torch.distributed.launch --nproc_per_node=8 train.py \
    --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gs6400/

# 그래디언트 누적 (더 큰 배치 시뮬레이션)
python train.py \
    --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gs6400/ \
    --gradient-accumulation 4

# 이전 체크포인트에서 재개
python train.py \
    --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gs6400/ \
    --resume-from out/gs6400/latest.pth
```

### 평가

```bash
# 단일 GPU 평가
python eval.py \
    --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gs6400/ \
    --resume-from out/gs6400/state_dict.pth

# 다중 GPU 평가
python -m torch.distributed.launch --nproc_per_node=8 eval.py \
    --py-config config/prob/nuscenes_gs6400.py \
    --work-dir out/gs6400/ \
    --resume-from out/gs6400/state_dict.pth
```

### 시각화

```bash
# 점유 예측 및 가우시안 시각화
CUDA_VISIBLE_DEVICES=0 python visualize.py \
    --py-config config/nuscenes_gs25600_solid.py \
    --work-dir out/nuscenes_gs25600_solid \
    --resume-from out/nuscenes_gs25600_solid/state_dict.pth \
    --vis-occ \
    --vis-gaussian \
    --num-samples 3 \
    --model-type base
```

### 데이터 준비

```bash
# 1. NuScenes 전체 데이터셋 다운로드
#    https://www.nuscenes.org/download

# 2. SurroundOcc 점유 어노테이션 다운로드
#    https://github.com/weiyithu/SurroundOcc

# 3. 카메라 정보 pkl 파일 다운로드
#    https://cloud.tsinghua.edu.cn/d/bb96379a3e46442c8898/

# 디렉터리 구조 구성
mkdir -p data/nuscenes data/nuscenes_cam data/surroundocc/samples
# 다운로드한 파일들을 위 경로에 배치
```

---

*분석 일자: 2026-02-19*
*저장소: /workspace/GaussianFormer*
