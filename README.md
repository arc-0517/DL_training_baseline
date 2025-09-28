# DL Training Baseline with Grad-CAM

딥러닝 모델 학습 및 추론을 위한 베이스라인 코드

## 📁 폴더 구조

```
DL_training_baseline/
├── data/
│   ├── dogcat_dataset/
│   └── custom_dataset/          # custom dataset 생성하면 됩니다.
├── dataloaders/
│   ├── __init__.py
│   └── datasets/
│       └── my_dataset.py
├── torch_trainer/
│   ├── models.py
│   └── trainer.py
├── utils/       
│   ├── __init__.py
│   └── gradcam.py
├── inference_folder/
├── inference_results/ 
├── configs.py         
├── main.py           
├── inference.py      
├── requirements.txt
└── README.md
```

## 🚀 사용법

### 1. 환경 설정

```bash
# Google Colab에서
!git clone https://github.com/arc-0517/DL_training_baseline.git
%cd DL_training_baseline

# 필요한 라이브러리 설치
!pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 폴더 구조 예시
data/
└── skin_dataset/
    ├── trainset/
    │   ├── benign/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── malignant/
    │       ├── image3.jpg
    │       └── image4.jpg
    └── testset/
        ├── benign/
        └── malignant/
```

### 3. 모델 학습

```bash
# EfficientNet-B0로 모델 학습
python main.py \
    --data_name skin \
    --model_name efficientnet_b0 \
    --n_class 2 \
    --epochs 50 \
    --batch_size 32 \
    --img_size 224

# ResNet18로 학습
python main.py \
    --data_name skin \
    --model_name resnet18 \
    --n_class 2 \
    --epochs 50

# 다른 EfficientNet 모델 사용
python main.py \
    --model_name efficientnet_b1 \
    --n_class 2
```

### 4. 추론 실행

```bash
# inference_folder에 추론할 이미지들 넣기
mkdir -p inference_folder
# 이미지 파일들을 inference_folder/에 복사

# 추론 실행 (Grad-CAM 포함)
python inference.py \
    --model_path ./save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00/model_last.pth \
    --model_name efficientnet_b0 \
    --n_class 2 \
    --class_names benign malignant \
    --generate_gradcam \
    --inference_folder ./inference_folder \
    --inference_output ./inference_results

# Grad-CAM 없이 추론만
python inference.py \
    --model_path ./path/to/model.pth \
    --model_name efficientnet_b0 \
    --n_class 2 \
    --class_names benign malignant
```

## 결과 확인

### 학습 완료 후 생성되는 파일들:
- `model_last.pth`: 최고 성능 모델
- `log.csv`: 학습 과정 로그
- `class_info.json`: 클래스 정보 및 모델 설정
- `configs.json`: 학습에 사용된 전체 설정

### 추론 완료 후 생성되는 파일들:
- `predictions.csv`: 예측 결과 (CSV 형식)
- `predictions.json`: 예측 결과 (JSON 형식)
- `gradcam/`: Grad-CAM 시각화 이미지들

### 예측 결과 예시:
```csv
filename,predicted_class,predicted_label,confidence,prob_benign,prob_malignant
image1.jpg,0,benign,0.892,0.892,0.108
image2.jpg,1,malignant,0.756,0.244,0.756
```

## 지원하는 모델

- **ResNet**: `resnet18`, `resnet50`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`

## Grad-CAM 기능

- 모든 지원 모델에서 Grad-CAM 생성 가능
- 예측 결과와 함께 시각적 설명 제공
- 원본 이미지, 히트맵, 오버레이 이미지를 한 번에 저장

## 주요 설정 옵션

```bash
# 모델 관련
--model_name: resnet18, resnet50, efficientnet_b0, efficientnet_b1, efficientnet_b2
--pre_trained: True/False (사전 훈련된 가중치 사용 여부)
--n_class: 클래스 수

# 데이터 관련  
--data_name: dogcat
--img_size: 입력 이미지 크기 (기본: 224)
--batch_size: 배치 크기

# 학습 관련
--epochs: 학습 에포크 수
--lr_ae: 학습률
--optimizer: adam, sgd, adamW

# 추론 관련
--generate_gradcam: Grad-CAM 생성 여부
--class_names: 클래스 이름들
```