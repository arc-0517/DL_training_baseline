# 피부 질환 분류 프로젝트

이 프로젝트는 지도 학습 기반 이미지 분류와 OpenAI의 CLIP 모델을 활용한 Zero-Shot 및 Few-Shot 이미지 분류를 수행합니다.

## 1. 환경 설정

### 가상 환경
conda 또는 venv와 같은 가상 환경 사용을 권장합니다.

```bash
conda create -n skin_clf python=3.9
conda activate skin_clf
```

### 의존성 설치
`requirements.txt` 파일에 명시된 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

CLIP 관련 기능만 사용한다면 `requirements_clip.txt`를 사용할 수 있습니다.
```bash
pip install -r requirements_clip.txt
```
만약 `clip` 설치에 문제가 발생하면 다음 명령어를 통해 직접 설치할 수 있습니다:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 데이터셋 구조
스크립트는 다음과 같은 데이터셋 디렉토리 구조를 예상합니다. 경로는 명령줄 인자를 통해 조정할 수 있습니다.

```
data/skin_dataset/
├── Training/
│   └── 01.원천데이터/
│       ├── acne/
│       │   ├── image1.png
│       │   └── ...
│       ├── atopic/
│       └── ...
└── Validation/
    └── 01.원천데이터/
        ├── acne/
        ├── atopic/
        └── ...
└── Test/
    └── 01.원천데이터/
        ├── acne/
        ├── atopic/
        └── ...
```

---

## 2. 지도 학습 기반 분류 (`main.py`)

`main.py` 스크립트는 표준적인 지도 학습 기반 이미지 분류 모델을 훈련하고 평가합니다. 다양한 모델 아키텍처, 데이터 증강, 손실 함수, 최적화 기법 등을 `configs.py`를 통해 유연하게 설정할 수 있습니다.

### 실행 방법
다음 명령어를 사용하여 `main.py`를 실행합니다.

```bash
python main.py [옵션들]
```

**기본 예시:**
ResNet18 모델을 사용하여 피부 데이터셋을 학습합니다.

```bash
python main.py --model_name resnet18 --data_name skin --epochs 50 --batch_size 64
```

**자주 사용되는 옵션:**

*   `--model_name`: 사용할 모델 아키텍처 (아래 **지원 모델** 참조)
*   `--data_name`: 데이터셋 이름 (예: `skin`, `dogcat`)
*   `--epochs`: 학습 에폭 수
*   `--batch_size`: 배치 크기
*   `--lr_ae`: 학습률
*   `--use_amp`: 자동 혼합 정밀도(Automatic Mixed Precision) 사용 여부 (학습 속도 향상)

### `configs.py`의 주요 파라미터 설명

`configs.py`는 `main.py`의 모든 설정 파라미터를 정의합니다. `argparse`를 기반으로 하며, 다음과 같은 주요 섹션으로 나뉩니다.

#### 2.1. Base Arguments (`base_parser`)

기본적인 설정과 로깅, 재현성 관련 옵션을 제어합니다.

*   `--checkpoint_root` (str, 기본값: `./save_results`): 모델 체크포인트 및 결과가 저장될 최상위 디렉토리.
*   `--random_state` (int, 기본값: `0`): 난수 시드. 재현성을 위해 중요합니다.
*   `--verbose` (bool, 기본값: `True`): 학습 과정 중 상세 로그 출력 여부.
*   `--confusion_matrix` (bool, 기본값: `True`): 평가 시 혼동 행렬(Confusion Matrix) 생성 여부.
*   `--wandb` (bool, 기본값: `True`): Weights & Biases 로깅 사용 여부.
*   `--wandb_project` (str, 기본값: `dogcat_classification`): Weights & Biases 프로젝트 이름.
*   `--wandb_name_tags` (list of str, 기본값: `['model_name', 'optimizer']`): Weights & Biases 런 이름에 포함될 태그 목록.

#### 2.2. Data Arguments (`data_parser`)

데이터 로딩, 전처리 및 증강 관련 옵션을 제어합니다.

*   `--data_dir` (str, 기본값: `./data`): 데이터셋의 루트 디렉토리.
*   `--data_name` (str, 기본값: `skin`, 선택: `dogcat`, `skin`): 사용할 데이터셋 이름.
*   `--valid_ratio` (float, 기본값: `0.2`): 훈련 데이터 중 검증 데이터로 할당할 비율.
*   `--shuffle_dataset` (bool, 기본값: `True`): 데이터셋 셔플 여부.
*   `--batch_size` (int, 기본값: `128`): 훈련 배치 크기.
*   `--test_batch_size` (int, 기본값: `64`): 테스트 배치 크기.
*   `--img_size` (int, 기본값: `224`): 이미지 리사이징 크기 (모든 이미지가 이 크기로 조정됩니다).
*   `--augmentation_type` (str, 기본값: `mixed`, 선택: `base`, `geometric`, `color`, `mixed`, `randaugment`, `autoaugment`): 사용할 데이터 증강 전략.
*   `--num_workers` (int, 기본값: `min(os.cpu_count(), 4)`): 데이터 로더 워커 프로세스 수.
*   `--selected_labels` (list of str, 기본값: `None`): 학습/검증에 사용할 특정 레이블 목록. `None`이면 모든 레이블 사용.

#### 2.3. Modeling Arguments (`modeling_parser`)

모델 아키텍처, 학습 하이퍼파라미터 및 손실 함수 관련 옵션을 제어합니다.

*   `--model_name` (str, 기본값: `resnet18`, **지원 모델** 참조): 사용할 모델 아키텍처.
*   `--pre_trained` (bool, 기본값: `True`): ImageNet 등으로 사전 학습된 가중치 사용 여부.
*   `--n_class` (int, 기본값: `6`): 분류할 클래스의 수.
*   `--loss_function` (str, 기본값: `ce`, 선택: `ce`, `mse`): 사용할 손실 함수 (Cross Entropy 또는 MSE).
*   `--optimizer` (str, 기본값: `adamW`, 선택: `adam`, `sgd`, `adamW`): 사용할 옵티마이저.
*   `--scheduler` (str, 기본값: `cosine`, 선택: `cosine`, `none`): 학습률 스케줄러.
*   `--lr_ae` (float, 기본값: `1e-3`): 초기 학습률.
*   `--weight_decay` (float, 기본값: `1e-4`): 가중치 감쇠(Weight Decay) 값.
*   `--epochs` (int, 기본값: `50`): 총 학습 에폭 수.
*   `--early_stopping_patience` (int, 기본값: `20`): Early Stopping을 위한 patience 값.
*   `--early_stopping_metric` (str, 기본값: `valid_loss`): Early Stopping 기준으로 사용할 메트릭.
*   `--use_amp` (bool, 기본값: `True`): Automatic Mixed Precision (AMP) 사용 여부.

**데이터 증강 및 손실 함수 관련:**

*   `--use_focal_loss` (bool, 기본값: `True`): Focal Loss 사용 여부.
*   `--focal_loss_gamma` (float, 기본값: `2.0`): Focal Loss의 감마(gamma) 값.
*   `--use_mixup` (bool, 기본값: `True`): Mixup 증강 기법 사용 여부.
*   `--mixup_alpha` (float, 기본값: `0.4`): Mixup의 알파(alpha) 값.
*   `--use_warmup` (bool, 기본값: `True`): Warm-up 스케줄러 사용 여부.
*   `--warmup_epochs` (int, 기본값: `5`): Warm-up 에폭 수.
*   `--label_smoothing` (float, 기본값: `0.1`): 레이블 스무딩(Label Smoothing) 강도.

#### 2.4. Inference Arguments (`inference_parser`)

추론(`inference.py`) 스크립트에서 사용되는 파라미터들입니다.

*   `--inference_folder` (str, 기본값: `./inference_folder`): 추론할 이미지가 있는 폴더.
*   `--inference_output` (str, 기본값: `./inference_results`): 추론 결과가 저장될 폴더.
*   `--model_path` (str, 기본값: `''`): 추론에 사용할 학습된 모델의 `.pth` 파일 경로.
*   `--generate_gradcam` (bool, 기본값: `True`): Grad-CAM 이미지 생성 여부.
*   `--class_names` (list of str, 기본값: `['seborrheic', 'rosacea', 'normal', 'acne', 'atopic', 'psoriasis']`): 분류할 클래스 이름 목록.

### 지원 모델

`--model_name` 인자에 사용할 수 있는 모델 목록입니다.

*   **ResNet 계열:** `resnet18`, `resnet50`
*   **EfficientNet 계열:** `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
*   **MobileNet 계열:** `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
*   **Vision Transformer (ViT) 계열:** `vit_b_16`, `vit_tiny_patch16_224`

---

## 3. 다중 실험 실행 (`run_experiments.py`)

`run_experiments.py` 스크립트는 `main.py`를 여러 번 반복적으로 실행하여 다양한 하이퍼파라미터 조합에 대한 실험을 자동화하는 유틸리티입니다. 이 스크립트를 통해 여러 모델, 증강 기법, 손실 함수 설정 등을 체계적으로 탐색할 수 있습니다.

### 실행 방법

스크립트 상단에 정의된 변수들을 수정하여 실험 조합을 설정한 후 실행합니다.

```bash
python run_experiments.py
```

**스크립트 내부 설정 예시:**

`run_experiments.py` 파일의 상단에서 `EFFICIENTNET_MODELS`, `AUGMENTATION_TYPES`, `MIXUP_OPTIONS`, `FOCAL_LOSS_OPTIONS` 등을 조절하여 원하는 실험을 구성할 수 있습니다.

```python
# 1. 모델 설정
EFFICIENTNET_MODELS = ['efficientnet_b1', 'efficientnet_b2']

# 3. Augmentation 설정
AUGMENTATION_TYPES = ['base', 'color', 'mixed', 'randaugment']

# 4. Mixup 적용 여부
MIXUP_OPTIONS = [True, False]

# 5. Loss 함수 선택 (Focal Loss 사용 여부)
FOCAL_LOSS_OPTIONS = [True, False]

# 7. 선택된 레이블 (None이면 전체 레이블 사용)
SELECTED_LABELS = ['normal', 'acne', 'rosacea', 'atopic']
```

---

## 4. CLIP Zero-Shot & Few-Shot 분류

이 섹션은 OpenAI의 CLIP 모델을 활용한 이미지 분류 방법을 설명합니다. 지도 학습과는 달리, 적은 양의 데이터로도 유연하게 분류 작업을 수행할 수 있습니다.

### 4.1. Zero-Shot Classification (`clip_zero_shot.py`)

CLIP Zero-Shot 분류는 사전 학습된 CLIP 모델을 사용하여 추가적인 학습 없이 텍스트 프롬프트와의 유사도를 기반으로 이미지를 분류합니다.

#### 실행 방법
`clip_zero_shot.py` 스크립트를 실행합니다.

**기본 예시:**
`ViT-B/32` 모델을 사용하여 검증 데이터셋의 모든 클래스에 대해 Zero-Shot 분류를 수행합니다.

```bash
python clip_zero_shot.py \
    --model_name "ViT-B/32" \
    --val_dir "data/skin_dataset/Validation/01.원천데이터"
```

**특정 클래스 지정 예시:**
`ViT-B/16` 모델을 사용하여 `acne`, `normal`, `rosacea` 클래스에 대해서만 Zero-Shot 분류를 수행합니다.

```bash
python clip_zero_shot.py \
    --model_name "ViT-B/16" \
    --classes acne normal rosacea \
    --val_dir "data/skin_dataset/Validation/01.원천데이터"
```

#### 주요 인자 (`clip_zero_shot.py`)
*   `--val_dir` (str, 기본값: `data/skin_dataset/Validation/01.원천데이터`): 검증 데이터 디렉토리 경로.
*   `--results_dir` (str, 기본값: `results/clip_zero_shot`): 결과 (JSON 및 혼동 행렬)를 저장할 디렉토리.
*   `--model_name` (str, 선택: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50`, `RN101`, 기본값: `ViT-B/32`): 사용할 CLIP 모델.
*   `--classes` (list of str, 선택 사항): 사용할 특정 클래스 이름 목록 (예: `acne atopic`). 제공되지 않으면 `val_dir` 내 모든 하위 디렉토리를 클래스로 사용.
*   `--batch_size` (int, 기본값: `32`): 특징 추출을 위한 배치 크기.
*   `--num_workers` (int, 기본값: `os.cpu_count() 또는 4 중 작은 값`): 데이터 로더를 위한 CPU 워커 수.

### 4.2. Few-Shot Classification (`clip_few_shot.py`)

이 방법은 훈련 데이터에서 추출한 소수의 레이블된 예시(support set)를 사용하여 모델을 미세 조정하거나 분류기를 학습시켜 검증 데이터(query set)에 대한 분류 성능을 향상시킵니다. K-NN (K-Nearest Neighbors)과 Linear Probe (MLP) 두 가지 방법을 지원합니다.

#### 실행 방법
`clip_few_shot.py` 스크립트를 실행하고 `--method` 인자로 원하는 방법을 지정합니다.

**예시 1: K-NN Few-Shot**
`ViT-B/32` 모델과 K-NN 분류기를 사용하여 10-shot 분류를 수행하며, `k=5`로 설정합니다.

```bash
python clip_few_shot.py \
    --method knn \
    --n_shot 10 \
    --k 5 \
    --model_name "ViT-B/32"
```

**예시 2: Linear Probe Few-Shot**
`ViT-B/32` 모델 위에 작은 MLP (Linear Probe)를 학습시켜 10-shot 분류를 수행합니다. 학습률(`--lr`)과 에폭(`--epochs`)을 지정합니다.

```bash
python clip_few_shot.py \
    --method linear_probe \
    --n_shot 10 \
    --model_name "ViT-B/32" \
    --lr 0.005 \
    --epochs 300
```

#### 주요 인자 (`clip_few_shot.py`)
*   `--method` (str, 선택: `knn`, `linear_probe`, 기본값: `linear_probe`): 사용할 Few-Shot 방법.
*   `--train_dir` (str, 기본값: `data/skin_dataset/Training/01.원천데이터`): 훈련 데이터 디렉토리 경로.
*   `--val_dir` (str, 기본값: `data/skin_dataset/Validation/01.원천데이터`): 검증 데이터 디렉토리 경로.
*   `--results_dir` (str, 기본값: `results/clip_few_shot`): 결과를 저장할 디렉토리.
*   `--model_name` (str, 기본값: `ViT-B/32`): 사용할 CLIP 모델.
*   `--classes` (list of str, 선택 사항): 사용할 특정 클래스 목록.
*   `--n_shot` (int, 기본값: `10`): Support Set을 위한 클래스당 샘플 수.
*   `--seed` (int, 기본값: `42`): 재현성을 위한 난수 시드.

**K-NN 전용 인자:**
*   `--k` (int, 기본값: `3`): K-NN 분류기에서 사용할 이웃의 수.

**Linear Probe 전용 인자:**
*   `--lr` (float, 기본값: `0.01`): Linear Probe 학습률.
*   `--weight_decay` (float, 기본값: `1e-4`): 옵티마이저의 가중치 감쇠.
*   `--epochs` (int, 기본값: `200`): Linear Probe 최대 학습 에폭 수.
*   `--hidden_dim` (int, 기본값: `256`): MLP의 은닉층 크기.
*   `--dropout` (float, 기본값: `0.3`): MLP의 드롭아웃 비율.
*   `--patience` (int, 기본값: `20`): Early Stopping의 patience 값.

---

## 5. 결과 (Results)
모든 실험 결과는 지정된 `--results_dir` 아래의 타임스탬프가 포함된 폴더에 저장됩니다. 각 실행 폴더에는 다음 파일이 포함됩니다:

*   `results_*.json`: 정확도, F1 점수, 정밀도, 재현율 등 상세 성능 메트릭이 포함된 JSON 파일.
*   `cm_*.png`: 혼동 행렬(Confusion Matrix) 이미지 파일.
*   `config.json` (지도 학습): 학습에 사용된 모든 설정 파라미터.
