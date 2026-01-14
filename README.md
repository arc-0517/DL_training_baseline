# 피부 질환 분류 딥러닝 프로젝트

PyTorch를 사용한 피부 질환 분류 모델 학습, 추론, 그리고 ONNX 모델 경량화를 위한 전체 파이프라인입니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 기능](#주요-기능)
- [폴더 구조](#폴더-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
  - [1. 데이터 준비](#1-데이터-준비)
  - [2. 모델 학습](#2-모델-학습)
  - [3. 모델 추론](#3-모델-추론)
  - [4. 모델 경량화 (ONNX)](#4-모델-경량화-onnx)
  - [5. 성능 벤치마크](#5-성능-벤치마크)
- [설정 파라미터 상세 가이드](#설정-파라미터-상세-가이드)
- [지원 모델](#지원-모델)
- [고급 기능](#고급-기능)
- [FAQ](#faq)
- [참고 자료](#참고-자료)

---

## 프로젝트 개요

이 프로젝트는 피부 질환 이미지를 분류하는 딥러닝 모델을 쉽게 학습하고 배포할 수 있도록 설계되었습니다.

### 주요 특징
- **다양한 CNN 모델 지원**: EfficientNet, ResNet, MobileNet, ViT 등
- **재현 가능한 학습**: 동일한 결과를 보장하는 시드 설정
- **Grad-CAM 시각화**: 모델 예측 결과를 시각적으로 설명
- **ONNX 모델 경량화**: 추론 속도를 향상
- **WandB 연동**: 학습 과정 실시간 모니터링
- **SegFormer 세그멘테이션**: 얼굴 영역 추출을 통한 정확도 향상

---

## 주요 기능

### 1. 모델 학습
- 사전 학습된 가중치 활용 (Transfer Learning)
- 다양한 데이터 증강 기법 (Base, Geometric, Color, Mixed, RandAugment, AutoAugment)
- 학습/검증/테스트 데이터 분할
- 조기 종료 (Early Stopping)
- Focal Loss 및 Mixup 지원
- Label Smoothing 및 Warm-up 스케줄러
- 학습 곡선 자동 저장

### 2. 모델 추론
- 단일 이미지 또는 폴더 단위 추론
- CSV/JSON 형식의 예측 결과 저장
- Grad-CAM을 통한 시각적 설명
- SegFormer 세그멘테이션 기반 피부 영역 추출
- 원본 vs 세그멘테이션 결과 비교

### 3. 모델 경량화
- PyTorch → ONNX 변환 (Opset 18 지원)
- 모델 크기 감소 및 추론 속도 향상 (CPU 기준 7배)
- 다양한 플랫폼 지원 (CPU, GPU, 모바일)
- 성능 벤치마크 도구 제공

---

## 폴더 구조

```
baseline_code/
├── data/                          # 데이터셋 디렉토리
│   ├── skin_dataset/              # 피부 질환 데이터셋
│   │   ├── trainset/              # 학습 데이터
│   │   │   ├── class1/            # 클래스 1
│   │   │   ├── class2/            # 클래스 2
│   │   │   └── ...
│   │   └── testset/               # 테스트 데이터
│   │       ├── class1/
│   │       ├── class2/
│   │       └── ...
│   └── dogcat_dataset/            # 예제 데이터셋
│
├── dataloaders/                   # 데이터 로더 모듈
│   ├── __init__.py
│   └── datasets/
│       └── my_dataset.py          # 커스텀 데이터셋 클래스
│
├── torch_trainer/                 # 학습 관련 모듈
│   ├── models.py                  # 모델 정의 및 빌더
│   ├── trainer.py                 # 학습 루프 구현
│   └── datasets.py                # 데이터셋 로더
│
├── utils/                         # 유틸리티 함수
│   ├── __init__.py
│   ├── gradcam.py                 # Grad-CAM 구현
│   ├── losses.py                  # 손실 함수
│   └── reproducibility.py         # 재현성 설정
│
├── save_results/                  # 학습된 모델 저장 디렉토리
│   └── dataset+skin/
│       └── model+efficientnet_b0/
│           └── 2024-01-01_12-00-00/
│               ├── model_last.pth    # 모델 가중치
│               ├── class_info.json   # 클래스 정보
│               ├── configs.json      # 학습 설정
│               └── log.csv           # 학습 로그
│
├── inference_set/                 # 추론할 이미지 폴더
├── inference_results/             # 추론 결과 저장 폴더
│   ├── predictions.csv            # 예측 결과 (CSV)
│   ├── predictions.json           # 예측 결과 (JSON)
│   └── gradcam/                   # Grad-CAM 이미지
│
├── onnx_models/                   # ONNX 모델 저장 디렉토리
│   └── clf_model.onnx             # 분류 모델 (ONNX)
│
├── configs.py                     # 설정 클래스 정의
├── main.py                        # 학습 메인 스크립트
├── inference.py                   # 추론 스크립트
├── inference_with_segmentation.py # 세그멘테이션 기반 추론
├── face_segmentation.py           # 얼굴 세그멘테이션 테스트
├── quantize_onnx_models.py        # ONNX 변환 스크립트
├── benchmark.py                   # 성능 벤치마크 스크립트
├── run_experiments.py             # 실험 자동화 스크립트
├── requirements.txt               # 필수 라이브러리 목록
└── README.md                      # 프로젝트 문서 (현재 파일)
```

---

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/arc-0517/DL_training_baseline.git -b skin_baseline
cd DL_training_baseline
```

### 2. 가상환경 생성 (권장)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 주요 라이브러리
- PyTorch >= 1.11.0
- torchvision
- timm (PyTorch Image Models)
- transformers (Hugging Face)
- onnx, onnxruntime
- opencv-python
- pillow
- pandas, numpy
- wandb (선택사항)

---

## 사용 방법

### 1. 데이터 준비

데이터는 다음과 같은 구조로 준비해야 합니다:

```
data/skin_dataset/
├── trainset/
│   ├── class1/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── class3/
│       └── ...
└── testset/
    ├── class1/
    ├── class2/
    └── class3/
```

#### 지원하는 이미지 형식
- JPG, JPEG, PNG, BMP

---

### 2. 모델 학습

#### 기본 사용법

```bash
python main.py \
    --data_name skin \
    --model_name efficientnet_b0 \
    --n_class 6 \
    --epochs 50 \
    --batch_size 32 \
    --img_size 224
```

#### 주요 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--data_name` | 데이터셋 이름 (skin, dogcat) | skin |
| `--model_name` | 사용할 모델 | efficientnet_b0 |
| `--n_class` | 클래스 개수 | 6 |
| `--epochs` | 학습 에포크 수 | 50 |
| `--batch_size` | 배치 크기 | 128 |
| `--img_size` | 입력 이미지 크기 | 224 |
| `--lr_ae` | 학습률 | 0.001 |
| `--optimizer` | 최적화 알고리즘 (adam, sgd, adamW) | adamW |
| `--pre_trained` | 사전 학습 가중치 사용 여부 | True |
| `--random_state` | 랜덤 시드 | 0 |

더 자세한 파라미터 설명은 [설정 파라미터 상세 가이드](#설정-파라미터-상세-가이드)를 참고하세요.

#### 다양한 모델로 학습

```bash
# ResNet18
python main.py --data_name skin --model_name resnet18 --n_class 6

# EfficientNet-B1
python main.py --data_name skin --model_name efficientnet_b1 --n_class 6

# ResNet50
python main.py --data_name skin --model_name resnet50 --n_class 6

# MobileNet V2 (경량 모델)
python main.py --data_name skin --model_name mobilenet_v2 --n_class 6
```

#### 학습 결과

학습이 완료되면 `save_results/` 디렉토리에 다음 파일들이 생성됩니다:

- `model_last.pth`: 학습된 모델 가중치
- `class_info.json`: 클래스 정보 및 모델 설정
- `configs.json`: 전체 학습 설정
- `log.csv`: 에포크별 학습 로그
- `confusion_matrix_test.csv`: 테스트 데이터 혼동 행렬

---

### 3. 모델 추론

#### 기본 추론

```bash
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --inference_dir inference_set \
    --output_dir inference_results
```

#### Grad-CAM 포함 추론

```bash
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --inference_dir inference_set \
    --output_dir inference_results \
    --generate_gradcam
```

#### 세그멘테이션 기반 추론

얼굴 영역만 추출하여 피부 질환을 분류합니다:

```bash
python inference_with_segmentation.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --inference_dir inference_set \
    --output_dir inference_results_segmented
```

#### SegFormer + Grad-CAM

```bash
python inference_with_segmentation.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --inference_dir inference_set \
    --output_dir inference_results_segmented \
    --generate_gradcam
```

#### 추론 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--model_path` | 학습된 모델 디렉토리 경로 | (필수) |
| `--inference_dir` | 추론할 이미지가 있는 폴더 | inference_set |
| `--output_dir` | 추론 결과 저장 폴더 | inference_results |
| `--generate_gradcam` | Grad-CAM 생성 여부 | False |

#### 추론 결과

추론이 완료되면 다음 파일들이 생성됩니다:

**predictions.csv**
```csv
filename,predicted_class,predicted_label,confidence,prob_class1,prob_class2,...
image1.jpg,0,benign,0.892,0.892,0.108
image2.jpg,1,malignant,0.756,0.244,0.756
```

**predictions.json**
```json
[
  {
    "filename": "image1.jpg",
    "predicted_class": 0,
    "predicted_label": "benign",
    "confidence": 0.892,
    "probabilities": {
      "benign": 0.892,
      "malignant": 0.108
    }
  }
]
```

---

### 4. 모델 경량화 (ONNX)

PyTorch 모델을 ONNX 형식으로 변환하여 추론 속도를 대폭 향상시킬 수 있습니다.

#### ONNX 변환

```bash
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --output_dir onnx_models
```

#### 세그멘테이션 모델 제외

인터넷 연결이 없거나 분류 모델만 필요한 경우:

```bash
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --output_dir onnx_models \
    --skip_segmentation
```

#### ONNX 변환 효과

- 모델 크기: 거의 동일 (약 0.3% 감소)
- **추론 속도: 약 7.4배 향상** (CPU 기준)
- 정확도: 동일 (정확도 손실 없음)
- 다양한 플랫폼 지원: CPU, GPU, 모바일 등

---

### 5. 성능 벤치마크

PyTorch와 ONNX 모델의 성능을 정량적으로 비교합니다.

#### 벤치마크 실행

```bash
python benchmark.py \
    --pytorch_model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --onnx_model_dir onnx_models
```

#### 세그멘테이션 모델 제외

```bash
python benchmark.py \
    --pytorch_model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --onnx_model_dir onnx_models \
    --skip_segmentation
```

#### 벤치마크 결과 예시

```
====================================================================
모델 성능 벤치마크: PyTorch vs ONNX
====================================================================
디바이스: cpu
반복 횟수: 100회

성능 비교:
        모델  프레임워크  크기 (MB)  평균 추론 시간 (ms)  표준편차 (ms) 속도 향상 크기 감소율
    분류 모델  PyTorch     15.32          46.5779        2.3781
    분류 모델     ONNX     15.27           6.3007        0.3631   7.39x    0.3%

====================================================================
요약
====================================================================

분류 모델:
  모델 크기:   15.32 MB → 15.27 MB (0.3% 감소)
  추론 시간:   46.5779 ms → 6.3007 ms (7.39x 향상)
====================================================================
```

---

## 설정 파라미터 상세 가이드

`configs.py` 파일에서 설정 가능한 모든 파라미터에 대한 상세한 설명입니다. 초보자도 쉽게 이해하고 수정할 수 있도록 작성되었습니다.

### 1. 기본 설정 (Base Parser)

학습 전반에 걸친 기본적인 설정입니다.

| 파라미터 | 타입 | 기본값 | 설명 | 수정 시기 |
|---------|------|--------|------|----------|
| `--checkpoint_root` | str | `./save_results` | 학습된 모델을 저장할 루트 디렉토리 경로 | 저장 위치를 변경하고 싶을 때 |
| `--random_state` | int | `0` | 랜덤 시드 값 (재현성을 위해 고정) | 다른 랜덤 초기화를 시도할 때 |
| `--verbose` | bool | `True` | 학습 중 상세 정보 출력 여부 | 출력을 줄이고 싶을 때 False |
| `--confusion_matrix` | bool | `True` | 혼동 행렬 저장 여부 | 혼동 행렬이 불필요할 때 False |
| `--wandb` | bool | `True` | WandB 로깅 사용 여부 | WandB를 사용하지 않으려면 False |
| `--wandb_project` | str | `dogcat_classification` | WandB 프로젝트 이름 | 프로젝트명을 변경하고 싶을 때 |
| `--wandb_name_tags` | list | `['model_name', 'optimizer']` | WandB 실행 이름에 포함할 태그 | 실행 이름 형식을 변경하고 싶을 때 |

**Tip!**
- `random_state`: 같은 값을 사용하면 매번 동일한 결과를 얻을 수 있습니다 (재현성)
- `wandb`: 첫 사용 시 `wandb login` 명령으로 로그인이 필요합니다

### 2. 데이터 설정 (Data Parser)

데이터셋 로딩 및 전처리 관련 설정입니다.

| 파라미터 | 타입 | 기본값 | 설명 | 수정 가이드 |
|---------|------|--------|------|-----------|
| `--data_dir` | str | `./data` | 데이터셋이 있는 디렉토리 경로 | 데이터셋 위치가 다를 때 수정 |
| `--data_name` | str | `skin` | 데이터셋 이름 (`skin`, `dogcat`) | 사용할 데이터셋에 맞게 선택 |
| `--valid_ratio` | float | `0.2` | 검증 데이터 비율 (0.0~1.0) | 검증 데이터 비율 조정 (권장: 0.1~0.3) |
| `--shuffle_dataset` | bool | `True` | 데이터셋 셔플 여부 | 일반적으로 True 유지 권장 |
| `--batch_size` | int | `128` | 학습 배치 크기 | GPU 메모리에 맞게 조정 (8, 16, 32, 64, 128) |
| `--test_batch_size` | int | `64` | 테스트 배치 크기 | 일반적으로 학습 배치의 절반 사용 |
| `--img_size` | int | `224` | 입력 이미지 크기 (정사각형) | 모델에 따라 224, 256, 384 등 사용 |
| `--augmentation_type` | str | `mixed` | 데이터 증강 타입 | 아래 증강 타입 설명 참고 |
| `--num_workers` | int | `min(cpu_count, 4)` | 데이터 로딩 워커 수 | CPU 코어 수에 맞게 자동 설정 |

**데이터 증강 타입 설명:**
- `base`: 기본 증강 (Resize, RandomCrop, Flip)
- `geometric`: 기하학적 변환 (회전, 이동, 확대/축소)
- `color`: 색상 변환 (밝기, 대비, 채도 조정)
- `mixed`: geometric + color 혼합 (권장)
- `randaugment`: 랜덤하게 증강 기법 조합
- `autoaugment`: 자동으로 최적 증강 기법 선택

**초보자 팁:**
- `batch_size`: GPU 메모리 부족 에러가 나면 값을 절반으로 줄이세요
- `img_size`: 224는 대부분의 모델에서 사용하는 표준 크기입니다
- `augmentation_type`: 처음에는 `mixed` 사용을 권장합니다

### 3. 모델 및 학습 설정 (Modeling Parser)

모델 아키텍처 및 학습 하이퍼파라미터 설정입니다.

#### 3.1 모델 아키텍처

| 파라미터 | 타입 | 기본값 | 설명 | 선택 가이드 |
|---------|------|--------|------|-----------|
| `--model_name` | str | `resnet18` | 사용할 모델 이름 | 아래 모델 선택 가이드 참고 |
| `--pre_trained` | bool | `True` | 사전 학습된 가중치 사용 여부 | True 권장 (Transfer Learning) |
| `--n_class` | int | `6` | 분류할 클래스 개수 | 데이터셋의 클래스 수와 일치시키기 |

**모델 선택 가이드:**

| 모델 이름 | 파라미터 수 | 추론 속도 | 정확도 | 추천 용도 |
|----------|-----------|---------|--------|---------|
| `mobilenet_v2` | 2.2M | 매우 빠름 | 보통 | 모바일/엣지 디바이스 |
| `mobilenet_v3_small` | 1.5M | 매우 빠름 | 보통 | 초경량 모델 필요 시 |
| `efficientnet_b0` | 5.3M | 빠름 | 높음 | **권장: 정확도와 속도 균형** |
| `efficientnet_b1` | 7.8M | 빠름 | 높음 | 더 높은 정확도 필요 시 |
| `resnet18` | 11.7M | 보통 | 높음 | 빠른 실험용 |
| `resnet50` | 25.6M | 느림 | 매우 높음 | 최고 정확도 필요 시 |

#### 3.2 손실 함수 및 최적화

| 파라미터 | 타입 | 기본값 | 설명 | 수정 가이드 |
|---------|------|--------|------|-----------|
| `--loss_function` | str | `ce` | 손실 함수 (`ce`: CrossEntropy, `mse`: MSE) | 분류 문제는 `ce` 사용 |
| `--optimizer` | str | `adamW` | 최적화 알고리즘 | 아래 최적화 알고리즘 설명 참고 |
| `--scheduler` | str | `cosine` | 학습률 스케줄러 (`cosine`, `none`) | `cosine` 권장 |
| `--lr_ae` | float | `0.001` | 초기 학습률 | 너무 크면 학습 불안정, 너무 작으면 느림 |
| `--weight_decay` | float | `0.0001` | 가중치 감쇠 (L2 정규화) | 과적합 방지 (0.0001~0.001) |

**최적화 알고리즘 설명:**
- `adam`: 가장 널리 사용되는 최적화 알고리즘
- `adamW`: Adam + Weight Decay (과적합 방지 강화, 권장)
- `sgd`: 전통적인 확률적 경사 하강법 (학습률 조정 필요)

**학습률 가이드:**
- `0.01`: 너무 큼 (학습 불안정)
- `0.001`: 일반적인 시작값 (권장)
- `0.0001`: 미세 조정 시

#### 3.3 학습 진행 설정

| 파라미터 | 타입 | 기본값 | 설명 | 수정 가이드 |
|---------|------|--------|------|-----------|
| `--epochs` | int | `50` | 전체 학습 에포크 수 | 데이터셋 크기에 따라 조정 |
| `--early_stopping_patience` | int | `10` | 조기 종료 인내 에포크 수 | 과적합 방지 (5~15 권장) |
| `--early_stopping_metric` | str | `valid_loss` | 조기 종료 기준 지표 | `valid_loss` 또는 `valid_acc` |
| `--use_amp` | bool | `True` | 자동 혼합 정밀도 (AMP) 사용 | GPU 메모리 절약 및 속도 향상 |

**에포크 수 가이드:**
- 소규모 데이터셋 (< 1,000장): 30~50 에포크
- 중규모 데이터셋 (1,000~10,000장): 50~100 에포크
- 대규모 데이터셋 (> 10,000장): 100+ 에포크

#### 3.4 고급 학습 기법

| 파라미터 | 타입 | 기본값 | 설명 | 사용 시기 |
|---------|------|--------|------|---------|
| `--use_focal_loss` | bool | `True` | Focal Loss 사용 여부 | 클래스 불균형 문제가 있을 때 |
| `--focal_loss_gamma` | float | `2.0` | Focal Loss 감마 값 | 불균형이 심할수록 큰 값 (1.0~5.0) |
| `--use_mixup` | bool | `True` | Mixup 데이터 증강 사용 | 일반화 성능 향상 (권장) |
| `--mixup_alpha` | float | `0.4` | Mixup 알파 값 | 0.2~1.0 사이 값 사용 |
| `--use_warmup` | bool | `True` | Warm-up 스케줄러 사용 | 학습 초반 안정성 향상 |
| `--warmup_epochs` | int | `5` | Warm-up 에포크 수 | 전체 에포크의 5~10% |
| `--label_smoothing` | float | `0.1` | Label Smoothing 값 | 과적합 방지 (0.0~0.2) |

**고급 기법 설명:**
- **Focal Loss**: 어려운 샘플에 더 집중하여 클래스 불균형 문제 해결
- **Mixup**: 두 이미지를 섞어서 학습하여 일반화 성능 향상
- **Warm-up**: 학습 초반에 작은 학습률로 시작하여 안정성 향상
- **Label Smoothing**: 정답 레이블을 부드럽게 만들어 과신 방지

### 4. 추론 설정 (Inference Parser)

추론 시 사용되는 설정입니다.

| 파라미터 | 타입 | 기본값 | 설명 | 수정 가이드 |
|---------|------|--------|------|-----------|
| `--inference_folder` | str | `./inference_folder` | 추론할 이미지가 있는 폴더 (사용 안 함) | inference.py에서 `--inference_dir` 사용 |
| `--inference_output` | str | `./inference_results` | 추론 결과 저장 폴더 (사용 안 함) | inference.py에서 `--output_dir` 사용 |
| `--model_path` | str | `''` | 학습된 모델 디렉토리 경로 | 추론 시 필수로 지정 |
| `--generate_gradcam` | bool | `True` | Grad-CAM 생성 여부 | 시각화가 필요 없으면 False |
| `--class_names` | list | `[class_0, ...]` | 클래스 이름 목록 | 일반적으로 자동 로드됨 |

**참고:** 추론 스크립트(`inference.py`)를 실행할 때는 다음 인자를 사용합니다:
- `--inference_dir`: 추론할 이미지 폴더
- `--output_dir`: 결과 저장 폴더
- `--generate_gradcam`: Grad-CAM 생성 여부

### 5. 실전 예제

#### 예제 1: 기본 학습 (초보자)

```bash
python main.py \
    --data_name skin \
    --model_name efficientnet_b0 \
    --n_class 6 \
    --epochs 50
```

#### 예제 2: 빠른 실험 (작은 모델, 적은 에포크)

```bash
python main.py \
    --data_name skin \
    --model_name mobilenet_v2 \
    --n_class 6 \
    --epochs 30 \
    --batch_size 64
```

#### 예제 3: 고정밀도 학습 (큰 모델, 많은 에포크)

```bash
python main.py \
    --data_name skin \
    --model_name resnet50 \
    --n_class 6 \
    --epochs 100 \
    --img_size 384 \
    --batch_size 32
```

#### 예제 4: 클래스 불균형 데이터셋

```bash
python main.py \
    --data_name skin \
    --model_name efficientnet_b0 \
    --n_class 6 \
    --use_focal_loss True \
    --focal_loss_gamma 3.0
```

#### 예제 5: 고급 증강 기법 사용

```bash
python main.py \
    --data_name skin \
    --model_name efficientnet_b1 \
    --n_class 6 \
    --augmentation_type randaugment \
    --use_mixup True \
    --mixup_alpha 0.8
```

#### 예제 6: GPU 메모리 절약

```bash
python main.py \
    --data_name skin \
    --model_name efficientnet_b0 \
    --n_class 6 \
    --batch_size 16 \
    --use_amp True
```

### 6. 문제 해결 가이드

#### GPU 메모리 부족 (Out of Memory)
- `--batch_size`를 절반으로 줄이기 (128 → 64 → 32)
- `--img_size`를 줄이기 (384 → 224)
- 더 작은 모델 사용 (resnet50 → efficientnet_b0 → mobilenet_v2)

#### 학습이 수렴하지 않음
- `--lr_ae`를 줄이기 (0.001 → 0.0001)
- `--use_warmup True` 설정
- `--warmup_epochs`를 늘리기 (5 → 10)

#### 과적합 (Overfitting)
- `--augmentation_type`을 `mixed` 또는 `randaugment`로 변경
- `--use_mixup True` 설정
- `--label_smoothing`을 증가 (0.1 → 0.2)
- `--weight_decay`를 증가 (0.0001 → 0.001)
- `--early_stopping_patience`를 줄이기 (10 → 5)

#### 학습 속도가 너무 느림
- `--batch_size`를 늘리기 (32 → 64 → 128)
- `--use_amp True` 설정 (GPU 사용 시)
- `--num_workers`를 늘리기 (CPU 코어 수에 맞게)

---

## 지원 모델

### 분류 모델 (Classification)

| 모델 이름 | 설명 | 파라미터 수 | 장점 |
|----------|------|-----------|------|
| `resnet18` | ResNet-18 | 11.7M | 빠른 실험 및 학습 |
| `resnet50` | ResNet-50 | 25.6M | 높은 정확도 |
| `efficientnet_b0` | EfficientNet-B0 | 5.3M | **권장: 속도와 정확도 균형** |
| `efficientnet_b1` | EfficientNet-B1 | 7.8M | 높은 정확도, 적당한 속도 |
| `efficientnet_b2` | EfficientNet-B2 | 9.2M | 더 높은 정확도 |
| `mobilenet_v2` | MobileNet V2 | 2.2M | 모바일/엣지 디바이스 최적화 |
| `mobilenet_v3_small` | MobileNet V3 Small | 1.5M | 초경량 모델 |
| `mobilenet_v3_large` | MobileNet V3 Large | 4.2M | 경량이면서 높은 정확도 |
| `vit_b_16` | Vision Transformer Base | 86M | Transformer 기반 (대규모 데이터) |
| `vit_tiny_patch16_224` | Vision Transformer Tiny | 5.7M | 경량 Transformer |

### 세그멘테이션 모델 (Segmentation)

| 모델 이름 | 설명 | 용도 |
|----------|------|------|
| SegFormer | Hugging Face 사전 학습 모델 | 얼굴 파싱/세그멘테이션 |

---

## 고급 기능

### 1. WandB 연동

학습 과정을 실시간으로 모니터링할 수 있습니다.

```bash
# WandB 설치 및 로그인
pip install wandb
wandb login

# 학습 시 자동으로 WandB에 로그 전송됨
python main.py --data_name skin --model_name efficientnet_b0
```

WandB를 사용하지 않으려면:
```bash
python main.py --wandb False
```

### 2. 실험 자동화

여러 모델을 자동으로 학습하고 비교할 수 있습니다.

```bash
python run_experiments.py
```

`run_experiments.py`를 수정하여 원하는 실험을 설정할 수 있습니다.

### 3. 재현 가능한 학습

동일한 결과를 보장하기 위해 시드를 고정합니다:

```bash
python main.py --random_state 42
```

`REPRODUCIBILITY.md` 파일에서 재현성 관련 자세한 내용을 확인할 수 있습니다.

### 4. Google Colab 사용

프로젝트에 포함된 Jupyter 노트북을 Google Colab에서 실행할 수 있습니다:

- `01_train_model.ipynb`: 모델 학습
- `02_inference.ipynb`: 추론 및 Grad-CAM
- `03_onnx_conversion.ipynb`: ONNX 변환
- `04_benchmark.ipynb`: 성능 벤치마크

각 노트북은 GitHub 저장소 클론, Google Drive 마운트, 데이터 로딩을 자동으로 처리합니다.

### 5. 커스텀 데이터셋 추가

새로운 데이터셋을 추가하려면 `dataloaders/datasets/my_dataset.py`를 수정하세요.

### 6. 새로운 모델 추가

`torch_trainer/models.py`에 새로운 모델을 추가할 수 있습니다.

---

## FAQ

### Q1. GPU 메모리 부족 에러가 발생합니다.

**A:** `--batch_size`를 줄이거나 `--img_size`를 줄이세요.
```bash
python main.py --batch_size 16 --img_size 224
```

### Q2. 학습이 너무 느립니다.

**A:** 다음을 시도해보세요:
- `--batch_size`를 늘리기
- `--use_amp True` 설정 (GPU 사용 시)
- 더 작은 모델 사용 (mobilenet_v2, efficientnet_b0)

### Q3. 추론 결과가 좋지 않습니다.

**A:** 다음을 확인하세요:
- 학습 데이터와 추론 이미지의 분포가 비슷한지
- 충분한 에포크로 학습했는지
- 데이터 증강이 적절한지
- SegFormer 세그멘테이션을 사용해보세요 (배경 제거)

### Q4. ONNX 변환 시 경고가 나옵니다.

**A:** `opset_version` 관련 경고는 무시해도 됩니다. 최종적으로 "✓ ONNX 모델 검증 완료"가 나오면 정상입니다.

### Q5. WandB 로그인이 안 됩니다.

**A:** WandB를 사용하지 않으려면:
```bash
python main.py --wandb False
```

---

## 참고 자료

### 논문
- **EfficientNet**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Grad-CAM**: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- **ONNX**: [ONNX: Open Neural Network Exchange](https://onnx.ai/)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- **Mixup**: [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

### 라이브러리
- [PyTorch](https://pytorch.org/)
- [timm (PyTorch Image Models)](https://github.com/rwightman/pytorch-image-models)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Weights & Biases](https://wandb.ai/)

### 도움이 되는 링크
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Transfer Learning 가이드](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ONNX 변환 가이드](https://pytorch.org/docs/stable/onnx.html)

---

## 시작하기

### 빠른 시작 가이드

```bash
# 1. 저장소 클론
git clone https://github.com/arc-0517/DL_training_baseline.git -b skin_baseline
cd DL_training_baseline

# 2. 환경 설정
pip install -r requirements.txt

# 3. 데이터 준비
# data/skin_dataset/ 디렉토리에 데이터 배치

# 4. 모델 학습
python main.py --data_name skin --model_name efficientnet_b0 --n_class 6 --epochs 50

# 5. 추론 실행
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/[날짜] \
    --inference_dir inference_set \
    --output_dir inference_results \
    --generate_gradcam

# 6. ONNX 변환
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/[날짜] \
    --skip_segmentation

# 7. 성능 비교
python benchmark.py \
    --pytorch_model_path save_results/dataset+skin/model+efficientnet_b0/[날짜] \
    --onnx_model_dir onnx_models \
    --skip_segmentation
```
