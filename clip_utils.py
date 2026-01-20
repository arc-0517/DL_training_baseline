"""
CLIP 실험을 위한 유틸리티 함수 및 클래스
"""

import torch
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class SkinDiseaseDataset(Dataset):
    """피부질환 이미지 데이터셋"""

    def __init__(self, root_dir, preprocess=None, selected_classes=None):
        """
        Args:
            root_dir: 데이터셋 루트 디렉토리 (e.g., "Training/01.원천데이터")
            preprocess: CLIP 전처리 함수
            selected_classes: 사용할 클래스 목록 (None이면 모든 클래스 사용)
        """
        self.root_dir = Path(root_dir)
        self.preprocess = preprocess
        
        # 하위 디렉토리를 클래스로 간주
        all_classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        if selected_classes:
            # provided class list 사용
            self.classes = sorted([c for c in all_classes if c in selected_classes])
        else:
            # 모든 클래스 사용
            self.classes = all_classes
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            # Support multiple image extensions
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in cls_dir.glob(ext):
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

        print(f"Found {len(self.samples)} images in {len(self.classes)} classes.")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.preprocess:
                image = self.preprocess(image)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}")
            # Return a dummy tensor and the label of the next sample to avoid crashing the loader
            return self.__getitem__((idx + 1) % len(self))
            
        return image, label, img_path


def extract_features(model, dataloader, device):
    """CLIP 모델을 사용하여 이미지 특징 추출"""
    all_features = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model.encode_image(images)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    return torch.cat(all_features).numpy(), np.array(all_labels), all_paths

def get_text_prompts(classes, custom_prompts=None):
    """
    주어진 클래스에 대한 텍스트 프롬프트 생성
    - 기본 템플릿 사용
    - 또는 사용자 정의 프롬프트 사용
    """
    if custom_prompts:
        return custom_prompts

    # 일반적인 템플릿
    templates = [
        "a photo of {} skin",
        "a dermatology image showing {}",
        "skin with {}",
        "a close-up of {} skin",
        "a photo of a person with {}",
        "{} skin condition",
    ]
    
    # 클래스 이름을 기반으로 프롬프트 생성
    prompts_per_class = {}
    for cls in classes:
        # Underscore and capitalize for better readability if needed
        pretty_cls = cls.replace('_', ' ').title()
        prompts_per_class[cls] = [template.format(pretty_cls) for template in templates]
        
    return prompts_per_class

def encode_text_prompts(model, prompts_dict, device):
    """
        텍스트 프롬프트를 인코딩하고, 프롬프트 앙상블(평균)을 수행
    """
    all_text_features = []
    
    with torch.no_grad():
        for cls, prompts in prompts_dict.items():
            text_tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 앙상블: 프롬프트 특징들의 평균 사용
            mean_features = text_features.mean(dim=0)
            mean_features /= mean_features.norm()
            
            all_text_features.append(mean_features)
            
    return torch.stack(all_text_features)

def evaluate(preds, labels, class_names, method_name, save_dir=None):
    """
    분류 결과를 평가하고, 결과를 저장
    - Accuracy, Precision, Recall, F1-score
    - Confusion Matrix
    """
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0, labels=range(len(class_names))
    )
    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    
    # 콘솔에 결과 출력
    print(f"\n--- Evaluation Results: {method_name} ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("\nPer-Class Performance:")
    for i, cls in enumerate(class_names):
        print(f"  - {cls:<15}: P={class_precision[i]:.3f}, R={class_recall[i]:.3f}, F1={class_f1[i]:.3f} (Support: {support[i]})")

    results = {
        'method': method_name,
        'accuracy': accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'per_class': {
            cls: {
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1': class_f1[i],
                'support': int(support[i])
            } for i, cls in enumerate(class_names)
        },
        'confusion_matrix': cm.tolist()
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 결과 저장
        json_path = save_dir / f"results_{method_name.replace(' ', '_').lower()}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {json_path}")
        
        # Confusion Matrix 시각화
        cm_path = save_dir / f"cm_{method_name.replace(' ', '_').lower()}.png"
        plot_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix - {method_name}")

    return results

def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Confusion Matrix를 시각화하고 파일로 저장"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")