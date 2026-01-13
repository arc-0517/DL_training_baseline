# 피부 질환 분류 딥러닝 프로젝트

PyTorch를 사용한 피부 질환 분류 모델 학습, 추론, 그리고 ONNX 모델 경량화를 위한 전체 파이프라인입니다.

## 📋 목차

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
- [지원 모델](#지원-모델)
- [고급 기능](#고급-기능)
- [FAQ](#faq)
- [참고 자료](#참고-자료)

---

## 🎯 프로젝트 개요

이 프로젝트는 피부 질환 이미지를 분류하는 딥러닝 모델을 쉽게 학습하고 배포할 수 있도록 설계되었습니다.

### 주요 특징
- ✅ **다양한 CNN 모델 지원**: EfficientNet, ResNet 등
- ✅ **재현 가능한 학습**: 동일한 결과를 보장하는 시드 설정
- ✅ **Grad-CAM 시각화**: 모델 예측 결과를 시각적으로 설명
- ✅ **ONNX 모델 경량화**: 추론 속도를 7배 이상 향상
- ✅ **WandB 연동**: 학습 과정 실시간 모니터링

---

## 🚀 주요 기능

### 1. 모델 학습
- 사전 학습된 가중치 활용 (Transfer Learning)
- 자동 데이터 증강 (Augmentation)
- 학습/검증/테스트 데이터 분할
- 조기 종료 (Early Stopping)
- 학습 곡선 자동 저장

### 2. 모델 추론
- 단일 이미지 또는 폴더 단위 추론
- CSV/JSON 형식의 예측 결과 저장
- Grad-CAM을 통한 시각적 설명
- 세그멘테이션 기반 피부 영역 추출

### 3. 모델 경량화
- PyTorch → ONNX 변환
- 모델 크기 감소 및 추론 속도 향상
- 다양한 플랫폼 지원 (CPU, GPU, 모바일)
- 성능 벤치마크 도구 제공

---

## 📁 폴더 구조

```
baseline_code/
├── data/                          # 데이터셋 디렉토리
│   ├── skin_dataset/              # 피부 질환 데이터셋
│   │   ├── trainset/              # 학습 데이터
│   │   │   ├── benign/            # 양성 클래스
│   │   │   └── malignant/         # 악성 클래스
│   │   └── testset/               # 테스트 데이터
│   └── dogcat_dataset/            # 예제 데이터셋
│
├── dataloaders/                   # 데이터 로더 모듈
│   ├── __init__.py
│   └── datasets/
│       └── my_dataset.py          # 커스텀 데이터셋 클래스
│
├── torch_trainer/                 # 학습 관련 모듈
│   ├── models.py                  # 모델 정의 및 빌더
│   └── trainer.py                 # 학습 루프 구현
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
│   ├── clf_model.onnx             # 분류 모델 (ONNX)
│   └── seg_model.onnx             # 세그멘테이션 모델 (ONNX)
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

## 💻 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd baseline_code
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

## 📖 사용 방법

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
| `--data_name` | 데이터셋 이름 | skin |
| `--model_name` | 사용할 모델 (resnet18, efficientnet_b0 등) | efficientnet_b0 |
| `--n_class` | 클래스 개수 | 2 |
| `--epochs` | 학습 에포크 수 | 50 |
| `--batch_size` | 배치 크기 | 32 |
| `--img_size` | 입력 이미지 크기 | 224 |
| `--lr_ae` | 학습률 | 0.001 |
| `--optimizer` | 최적화 알고리즘 (adam, sgd, adamW) | adam |
| `--pre_trained` | 사전 학습 가중치 사용 여부 | True |
| `--random_state` | 랜덤 시드 | 42 |

#### 다양한 모델로 학습

```bash
# ResNet18
python main.py --data_name skin --model_name resnet18 --n_class 6

# EfficientNet-B1
python main.py --data_name skin --model_name efficientnet_b1 --n_class 6

# ResNet50
python main.py --data_name skin --model_name resnet50 --n_class 6
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
    --inference_folder inference_set \
    --inference_output inference_results
```

#### Grad-CAM 포함 추론

```bash
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --inference_folder inference_set \
    --inference_output inference_results \
    --generate_gradcam
```

#### 세그멘테이션 기반 추론

얼굴 영역만 추출하여 피부 질환을 분류합니다:

```bash
python inference_with_segmentation.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --inference_folder inference_set \
    --inference_output inference_results_segmented
```

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
- 다양한 플랫폼 지원: CPU, GPU, 모바일 등

---

### 5. 성능 벤치마크

PyTorch와 ONNX 모델의 성능을 정량적으로 비교합니다.

```bash
python benchmark.py \
    --pytorch_model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
    --onnx_model_dir onnx_models
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

## 🤖 지원 모델

### 분류 모델 (Classification)

| 모델 이름 | 설명 | 파라미터 수 |
|----------|------|-----------|
| `resnet18` | ResNet-18 | 11.7M |
| `resnet50` | ResNet-50 | 25.6M |
| `efficientnet_b0` | EfficientNet-B0 | 5.3M |
| `efficientnet_b1` | EfficientNet-B1 | 7.8M |
| `efficientnet_b2` | EfficientNet-B2 | 9.2M |

### 세그멘테이션 모델 (Segmentation)

| 모델 이름 | 설명 | 용도 |
|----------|------|------|
| SegFormer | Hugging Face 사전 학습 모델 | 얼굴 파싱/세그멘테이션 |

---

## 🎓 고급 기능

### 1. WandB 연동

학습 과정을 실시간으로 모니터링할 수 있습니다.

```bash
# WandB 설치 및 로그인
pip install wandb
wandb login

# 학습 시 자동으로 WandB에 로그 전송됨
python main.py --data_name skin --model_name efficientnet_b0
```

### 2. 실험 자동화

여러 모델을 자동으로 학습하고 비교할 수 있습니다.

```bash
python run_experiments.py
```

`run_experiments_template.py`를 수정하여 원하는 실험을 설정할 수 있습니다.

### 3. 재현 가능한 학습

동일한 결과를 보장하기 위해 시드를 고정합니다:

```bash
python main.py --random_state 42
```

`REPRODUCIBILITY.md` 파일에서 재현성 관련 자세한 내용을 확인할 수 있습니다.

### 4. 커스텀 데이터셋 추가

`dataloaders/datasets/my_dataset.py`를 수정하여 새로운 데이터셋을 추가할 수 있습니다.

```python
def make_data_loaders(config):
    if config.data_name == 'my_dataset':
        # 커스텀 데이터셋 로더 구현
        pass
```

### 5. 새로운 모델 추가

`torch_trainer/models.py`에 새로운 모델을 추가할 수 있습니다.

```python
def build_model(model_name, pre_trained=True, n_class=2):
    if model_name == 'my_custom_model':
        model = MyCustomModel(num_classes=n_class)
        return model
```

---

## ❓ FAQ

### Q1. GPU를 사용할 수 없나요?
**A:** PyTorch가 CUDA를 지원하도록 설치되어 있고 NVIDIA GPU가 있다면 자동으로 GPU를 사용합니다.

```bash
# GPU 사용 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### Q2. 클래스 개수를 어떻게 설정하나요?
**A:** `--n_class` 파라미터로 설정합니다. 데이터셋의 하위 폴더 개수와 일치해야 합니다.

```bash
python main.py --n_class 6  # 6개 클래스
```

### Q3. 메모리 부족 오류가 발생합니다.
**A:** 배치 크기를 줄이거나 이미지 크기를 조정하세요.

```bash
python main.py --batch_size 16 --img_size 192
```

### Q4. ONNX 변환이 실패합니다.
**A:** `onnx`와 `onnxruntime` 패키지가 설치되어 있는지 확인하세요.

```bash
pip install onnx onnxruntime
```

### Q5. 학습 중에 과적합이 발생합니다.
**A:** 다음 방법들을 시도해보세요:
- 데이터 증강 강화
- Dropout 추가
- Learning rate 조정
- Early stopping 활용
- 정규화 (L2 regularization) 추가

---

## 📚 참고 자료

### 논문
- **EfficientNet**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Grad-CAM**: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- **ONNX**: [ONNX: Open Neural Network Exchange](https://onnx.ai/)

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

## 📝 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.

---

## 💬 문의사항

문제가 발생하거나 질문이 있으시면 다음을 확인해주세요:

1. 이 README 파일의 FAQ 섹션
2. `REPRODUCIBILITY.md` 파일 (재현성 관련)
3. 각 스크립트 파일의 docstring 및 주석

---

## 🎉 시작하기

### 빠른 시작 가이드

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 준비
# data/skin_dataset/ 디렉토리에 데이터 배치

# 3. 모델 학습
python main.py --data_name skin --model_name efficientnet_b0 --n_class 6 --epochs 50

# 4. 추론 실행
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/[날짜] \
    --inference_folder inference_set \
    --generate_gradcam

# 5. ONNX 변환
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/[날짜]

# 6. 성능 비교
python benchmark.py \
    --pytorch_model_path save_results/dataset+skin/model+efficientnet_b0/[날짜]
```

행복한 딥러닝 하세요! 🚀
