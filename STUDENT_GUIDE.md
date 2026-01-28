# 피부 질환 분류 시스템 - 학생용 가이드

이 가이드는 피부 질환 분류 시스템의 각 기능별 사용법과 파라미터 설명을 담고 있습니다.

---

## 목차

1. [모델 학습 (Training)](#1-모델-학습-training)
2. [반복 실험 (Experiments)](#2-반복-실험-experiments)
3. [모델 추론 (Inference)](#3-모델-추론-inference)
4. [피부 영역 추출 (Segmentation)](#4-피부-영역-추출-segmentation)
5. [모델 경량화 (ONNX 변환)](#5-모델-경량화-onnx-변환)
6. [API 서버 실행](#6-api-서버-실행)

---

## 1. 모델 학습 (Training)

### 기본 실행

```bash
python main.py --data_name skin --model_name efficientnet_b0 --epochs 50
```

### 주요 파라미터

#### 기본 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--checkpoint_root` | `./save_results` | 모델 저장 경로 |
| `--random_state` | `0` | 랜덤 시드 (재현성 보장) |
| `--wandb` | `True` | Weights & Biases 로깅 사용 여부 |
| `--wandb_project` | `dogcat_classification` | W&B 프로젝트 이름 |

#### 데이터 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--data_dir` | `./data` | 데이터셋 경로 |
| `--data_name` | `skin` | 데이터셋 이름 (`skin`, `dogcat`) |
| `--batch_size` | `128` | 배치 크기 (GPU 메모리에 따라 조정) |
| `--img_size` | `224` | 입력 이미지 크기 |
| `--valid_ratio` | `0.2` | 검증 데이터 비율 (20%) |
| `--augmentation_type` | `mixed` | 데이터 증강 방식 |
| `--selected_labels` | `None` | 사용할 레이블 선택 (미지정 시 전체) |

**augmentation_type 옵션:**
- `base`: 기본 변환만 (Resize, Normalize)
- `geometric`: 기하학적 변환 (회전, 뒤집기)
- `color`: 색상 변환 (밝기, 대비, 채도)
- `mixed`: 기하학적 + 색상 변환 조합
- `randaugment`: RandAugment 적용
- `autoaugment`: AutoAugment 적용

#### 모델 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model_name` | `resnet18` | 사용할 모델 아키텍처 |
| `--pre_trained` | `True` | ImageNet 사전학습 가중치 사용 |
| `--n_class` | `6` | 분류할 클래스 수 |

**사용 가능한 모델:**
- ResNet: `resnet18`, `resnet50`
- EfficientNet: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- MobileNet: `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
- ViT: `vit_b_16`, `vit_tiny_patch16_224`

#### 학습 설정

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | `50` | 총 학습 에폭 수 |
| `--lr_ae` | `0.001` | 학습률 (Learning Rate) |
| `--optimizer` | `adamW` | 옵티마이저 (`adam`, `sgd`, `adamW`) |
| `--scheduler` | `cosine` | 학습률 스케줄러 (`cosine`, `none`) |
| `--weight_decay` | `0.0001` | 가중치 감쇠 (L2 정규화) |
| `--early_stopping_patience` | `20` | 조기 종료 인내심 (에폭 수) |

#### 고급 학습 기법

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--use_mixup` | `True` | Mixup 데이터 증강 사용 |
| `--mixup_alpha` | `0.4` | Mixup 혼합 비율 |
| `--use_focal_loss` | `True` | Focal Loss 사용 (불균형 데이터에 효과적) |
| `--focal_loss_gamma` | `2.0` | Focal Loss 감마 값 |
| `--label_smoothing` | `0.1` | 레이블 스무딩 비율 |
| `--use_warmup` | `True` | 학습률 워밍업 사용 |
| `--warmup_epochs` | `5` | 워밍업 에폭 수 |
| `--use_amp` | `True` | Mixed Precision Training 사용 |

### 예시: 다양한 설정으로 학습

```bash
# EfficientNet-B1으로 학습, 배치 크기 64
python main.py \
    --data_name skin \
    --model_name efficientnet_b1 \
    --batch_size 64 \
    --epochs 100 \
    --lr_ae 0.0005

# 특정 레이블만 사용하여 학습
python main.py \
    --data_name skin \
    --model_name efficientnet_b0 \
    --selected_labels normal acne rosacea atopic

# Mixup과 Focal Loss 비활성화
python main.py \
    --data_name skin \
    --model_name resnet50 \
    --use_mixup False \
    --use_focal_loss False
```

### 학습 결과 저장 위치

```
save_results/
└── dataset+skin/
    └── model+efficientnet_b0/
        └── 2026-01-28_14-30-00/
            ├── model_last.pth      # 모델 가중치
            ├── class_info.json     # 클래스 정보 및 설정
            ├── results.json        # 학습 결과
            └── confusion_matrix.png # 혼동 행렬
```

---

## 2. 반복 실험 (Experiments)

여러 하이퍼파라미터 조합을 자동으로 실험합니다.

### 실행 방법

```bash
python run_experiments.py
```

### 설정 변경 (run_experiments.py 파일 수정)

```python
# 실험할 모델 목록
EFFICIENTNET_MODELS = ['efficientnet_b1', 'efficientnet_b2']

# 배치 크기
BATCH_SIZE = 64

# 데이터 증강 방식
AUGMENTATION_TYPES = ['base', 'color', 'mixed', 'randaugment']

# Mixup 사용 여부
MIXUP_OPTIONS = [True, False]

# Focal Loss 사용 여부
FOCAL_LOSS_OPTIONS = [True, False]

# 학습 설정
LEARNING_RATE = 0.001
EPOCHS = 50
RANDOM_STATE = 42

# 사용할 레이블 (None이면 전체 사용)
SELECTED_LABELS = ['normal', 'acne', 'rosacea', 'atopic']
```

### 실험 조합 계산

위 설정의 경우:
- 모델: 2개 × 증강: 4개 × Mixup: 2개 × Focal Loss: 2개
- **총 32개 실험** 자동 실행

---

## 3. 모델 추론 (Inference)

학습된 모델로 새 이미지를 예측합니다.

### 기본 실행

```bash
python inference.py --model_path save_results/dataset+skin/model+efficientnet_b0/2026-01-28_14-30-00
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model_path` | (필수) | 학습된 모델 디렉토리 경로 |
| `--inference_dir` | `inference_set` | 추론할 이미지가 있는 폴더 |
| `--output_dir` | `inference_results` | 결과 저장 폴더 |
| `--generate_gradcam` | `False` | Grad-CAM 시각화 생성 |

### 예시

```bash
# Grad-CAM 시각화 포함
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b1/2026-01-28_14-30-00 \
    --inference_dir my_test_images \
    --output_dir my_results \
    --generate_gradcam

# 기본 추론만
python inference.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2026-01-28_14-30-00
```

### 결과 파일

```
inference_results/
├── inference_results.json    # 예측 결과 (JSON)
└── gradcam/                  # Grad-CAM 시각화 이미지
    ├── gradcam_image1.png
    └── gradcam_image2.png
```

### Grad-CAM이란?

Grad-CAM(Gradient-weighted Class Activation Mapping)은 모델이 **이미지의 어느 부분을 보고 판단했는지** 시각화하는 기법입니다.
- 빨간색: 모델이 중요하게 본 영역
- 파란색: 덜 중요한 영역

---

## 4. 피부 영역 추출 (Segmentation)

SegFormer 모델을 사용하여 얼굴 이미지에서 피부 영역만 추출합니다.

### Google Colab에서 실행

`face_segmentation.py` 파일을 Colab에서 실행하거나, 아래 코드를 사용하세요.

### 핵심 코드

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np

# 모델 로드
processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.eval()

# 이미지 로드
image = Image.open("face_image.jpg").convert("RGB")

# 추론
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 세그멘테이션 맵 생성
logits = outputs.logits
upsampled = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
parsing_map = upsampled.argmax(dim=1)[0].numpy()

# 피부만 추출 (라벨 1 = 피부)
img_array = np.array(image).copy()
skin_mask = (parsing_map == 1)
img_array[~skin_mask] = 0  # 피부 외 영역을 검은색으로

skin_only = Image.fromarray(img_array)
skin_only.save("skin_only.png")
```

### 세그멘테이션 라벨

| 라벨 ID | 영역 |
|---------|------|
| 0 | 배경 (background) |
| 1 | **피부 (skin)** |
| 2 | 눈썹 (eyebrow) |
| 3 | 눈 (eye) |
| 4 | 안경 (glasses) |
| 5 | 귀 (ear) |
| 6 | 코 (nose) |
| 7 | 입 (mouth) |
| 8 | 목 (neck) |
| 9 | 머리카락 (hair) |

---

## 5. 모델 경량화 (ONNX 변환)

PyTorch 모델을 ONNX 형식으로 변환하여 추론 속도를 최적화합니다.

### 기본 실행

```bash
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2026-01-28_14-30-00
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--model_path` | (필수) | 학습된 모델 디렉토리 경로 |
| `--output_dir` | `onnx_models` | ONNX 모델 저장 경로 |
| `--skip_segmentation` | `False` | 세그멘테이션 모델 변환 건너뛰기 |

### 예시

```bash
# 분류 모델만 변환 (세그멘테이션 제외)
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b0/2026-01-28_14-30-00 \
    --skip_segmentation

# 출력 디렉토리 지정
python quantize_onnx_models.py \
    --model_path save_results/dataset+skin/model+efficientnet_b1/2026-01-28_14-30-00 \
    --output_dir my_onnx_models
```

### 변환 결과

```
onnx_models/
├── clf_model.onnx       # 분류 모델 (EfficientNet 등)
└── seg_model.onnx       # 세그멘테이션 모델 (SegFormer)
```

### ONNX의 장점

1. **빠른 추론 속도**: CPU에서도 빠르게 동작
2. **크로스 플랫폼**: Windows, Linux, Mac, 모바일 등 다양한 환경 지원
3. **경량화**: 모델 크기 감소

---

## 6. API 서버 실행

학습된 모델을 웹 API로 서빙합니다.

### 기본 실행 (ONNX 모델 사용)

```bash
python api.py
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--host` | `0.0.0.0` | 서버 호스트 |
| `--port` | `8000` | 서버 포트 |
| `--model_path` | `None` | PyTorch 모델 경로 (미지정 시 ONNX 사용) |
| `--watch` | `False` | 폴더 감시 모드 활성화 |

### 실행 예시

```bash
# 기본 실행 (ONNX 모델)
python api.py

# PyTorch 모델 사용
python api.py --model_path save_results/dataset+skin/model+efficientnet_b1/2026-01-28_14-30-00

# 포트 변경
python api.py --port 8080

# 폴더 감시 모드 (API + 자동 처리)
python api.py --watch

```

### API 엔드포인트

| URL | 메서드 | 설명 |
|-----|--------|------|
| `/` | GET | API 상태 확인 |
| `/docs` | GET | **Swagger UI** (API 테스트 페이지) |
| `/model/info` | GET | 현재 모델 정보 |
| `/segment` | POST | 피부 영역 세그멘테이션 |
| `/predict` | POST | 피부 상태 예측 |
| `/analyze` | POST | **세그멘테이션 + 예측 통합** |

### Swagger UI에서 테스트

1. 브라우저에서 `http://localhost:8000/docs` 접속
2. `/analyze` 클릭
3. "Try it out" 클릭
4. 이미지 파일 선택
5. "Execute" 클릭
6. 결과 확인

### 폴더 감시 모드

| 폴더 | 용도 |
|------|------|
| `watch_input/` | 이미지를 넣으면 자동 처리 |
| `watch_output/` | 처리 결과 저장 |

```bash
# 폴더 감시 모드 실행
python api.py --watch

# watch_input/ 폴더에 이미지 복사하면 자동 처리됨
# 결과는 watch_output/ 폴더에 저장
```

---

## 전체 워크플로우

```
1. 데이터 준비
   └── data/skin_dataset/ 폴더에 이미지 배치

2. 모델 학습
   └── python main.py --data_name skin --model_name efficientnet_b0

3. (선택) 반복 실험
   └── python run_experiments.py

4. 모델 추론 테스트
   └── python inference.py --model_path <모델경로> --generate_gradcam

5. 모델 경량화
   └── python quantize_onnx_models.py --model_path <모델경로>

6. API 서버 배포
   └── python api.py
```

---

## 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Weights & Biases](https://wandb.ai/)
