# inference.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
import argparse
import glob
import numpy as np
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

from configs import InferenceConfig
from torch_trainer.models import build_model, get_gradcam_target_layer
from utils.gradcam import GradCAM, save_gradcam_result

class InferenceDataset(Dataset):
    """추론용 데이터셋 클래스"""
    def __init__(self, data_path, transform=None, img_size=224):
        self.data_path = data_path
        self.transform = transform
        self.img_size = img_size
        self.image_paths = []
        
        # 이미지 파일들 수집
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        # 폴더 내 모든 파일 검사
        if os.path.exists(data_path):
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if os.path.isfile(file_path):                    
                    _, ext = os.path.splitext(filename.lower())
                    if ext in image_extensions:
                        self.image_paths.append(file_path)
        
        # 중복 제거 및 정렬
        self.image_paths = sorted(list(set(self.image_paths)))
        
        print(f"Found {len(self.image_paths)} images in {data_path}")                
        
        # 원본 이미지용 변환
        self.original_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 에러: {e}")            
            image = Image.new('RGB', (224, 224), color='black')
        
        # 원본 이미지
        original_image = self.original_transform(image)
        
        # 모델 입력용 이미지
        if self.transform:
            image = self.transform(image)
        else:
            image = self.original_transform(image)
            
        return image, original_image, image_path

def load_model(model_path, model_name, n_class, device):
    """모델 로드"""
    model = build_model(model_name=model_name, pre_trained=False, n_class=n_class)
    
    # 체크포인트 로드
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"모델 로드 성공: {model_path}")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("CPU로 다시 시도합니다...")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"CPU로 모델 로드 성공")
    
    try:
        model.to(device)
        print(f"모델이 {device}로 이동 완료")
    except Exception as e:
        print(f"모델 디바이스 이동 실패: {e}")
        device = torch.device('cpu')
        model.to(device)
        print(f"강제로 CPU 모드로 전환")
    
    model.eval()
    return model

def get_transforms(img_size=224):
    """이미지 전처리 변환"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def run_inference(config):
    """추론 실행"""
    # CUDA 사용 가능 여부 확인 및 디바이스 설정
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            # 실제로 CUDA 디바이스가 작동하는지 테스트
            test_tensor = torch.tensor([1.0]).to(device)
            print(f"CUDA 사용 가능: {device}")
            print(f"GPU: {torch.cuda.get_device_name()}")
        except Exception as e:
            print(f"CUDA 초기화 실패: {e}")
            print("CPU 모드로 전환합니다.")
            device = torch.device('cpu')
    else:
        print("CUDA를 사용할 수 없습니다.")
        print("CPU 모드로 추론을 진행합니다.")
        device = torch.device('cpu')
    
    print(f"사용 디바이스: {device}")
    
    # 출력 폴더 생성
    os.makedirs(config.inference_output, exist_ok=True)
    gradcam_output = os.path.join(config.inference_output, 'gradcam')
    if config.generate_gradcam:
        os.makedirs(gradcam_output, exist_ok=True)
    
    # 모델 로드
    print(f"Loading model from: {config.model_path}")
    model = load_model(config.model_path, config.model_name, config.n_class, device)
    
    # 데이터셋 및 데이터로더 생성
    transform = get_transforms(config.img_size)
    dataset = InferenceDataset(config.inference_folder, transform=transform, img_size=config.img_size)
    
    if len(dataset) == 0:
        print("추론할 이미지가 없습니다. inference_folder에 이미지를 넣어주세요.")
        return
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # Grad-CAM 설정
    gradcam = None
    if config.generate_gradcam:
        target_layer = get_gradcam_target_layer(model, config.model_name)
        gradcam = GradCAM(model, target_layer)
    
    # 추론 결과 저장용
    results = []
    
    print("Starting inference...")
    for batch_idx, (images, original_images, image_paths) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # 배치 내 각 이미지 처리
        for i in range(len(images)):
            image_path = image_paths[i]
            filename = os.path.basename(image_path)
            predicted_class = predictions[i].item()
            confidence = probs[i][predicted_class].item()
            
            # 결과 저장
            result = {
                'filename': filename,
                'predicted_class': predicted_class,
                'predicted_label': config.class_names[predicted_class],
                'confidence': confidence
            }
            
            # 클래스별 확률 추가
            for j, class_name in enumerate(config.class_names):
                result[f'prob_{class_name}'] = probs[i][j].item()
            
            results.append(result)
            
            # Grad-CAM 생성 및 저장
            if config.generate_gradcam:
                single_image = images[i:i+1]  # 단일 이미지로 배치 유지
                cam, pred_class, pred_conf = gradcam.generate_cam(single_image, predicted_class)
                
                # 원본 이미지 (tensor)
                original_img = original_images[i]  # (C, H, W) tensor
                overlayed, heatmap = gradcam.visualize_cam(original_img, cam)
                
                # Grad-CAM 결과 저장
                gradcam_filename = f"{os.path.splitext(filename)[0]}_gradcam.png"
                gradcam_path = os.path.join(gradcam_output, gradcam_filename)
                save_gradcam_result(
                    original_img, (overlayed, heatmap), gradcam_path,
                    predicted_class, confidence, config.class_names
                )
    
    # 결과를 CSV로 저장
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(config.inference_output, 'predictions.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    # JSON으로도 저장
    results_json_path = os.path.join(config.inference_output, 'predictions.json')
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference completed!")
    print(f"Results saved to: {results_csv_path}")
    print(f"Total images processed: {len(results)}")
    
    # 예측 결과 요약 출력
    print("\nPrediction Summary:")
    for class_name in config.class_names:
        count = sum(1 for r in results if r['predicted_label'] == class_name)
        print(f"  {class_name}: {count} images")

def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--inference_folder', type=str, default='./inference_folder',
                       help='Folder containing images for inference')
    parser.add_argument('--inference_output', type=str, default='./inference_results',
                       help='Output folder for results')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       help='Model architecture name')
    parser.add_argument('--n_class', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--generate_gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--class_names', nargs='+', default=['negative', 'positive'],
                       help='Class names')
    
    args = parser.parse_args()
    
    # InferenceConfig 객체 생성
    config = InferenceConfig(**vars(args))
    
    # 추론 실행
    run_inference(config)

if __name__ == '__main__':
    main()