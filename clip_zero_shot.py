"""
CLIP Zero-Shot Classification
- 이미지 인코더와 텍스트 인코더의 유사도를 사용하여 분류
- 사전 학습된 CLIP 모델만 사용 (별도 학습 없음)
"""

import os
import torch
import clip
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from clip_utils import (
    SkinDiseaseDataset,
    extract_features,
    get_text_prompts,
    encode_text_prompts,
    evaluate
)

def main(args):
    """메인 실행 함수"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 결과 저장 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.results_dir) / f"{args.model_name.replace('/', '-')}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLIP Zero-Shot Classification")
    print(f"  - Model: {args.model_name}")
    print(f"  - Validation Dir: {args.val_dir}")
    print(f"  - Classes: {'All' if not args.classes else args.classes}")
    print(f"  - Results will be saved to: {save_dir}")
    print("=" * 70)

    # CLIP 모델 로드
    model, preprocess = clip.load(args.model_name, device=device)
    model.eval()

    # 데이터셋 및 데이터로더
    val_dataset = SkinDiseaseDataset(
        root_dir=args.val_dir,
        preprocess=preprocess,
        selected_classes=args.classes
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 클래스 이름 가져오기
    class_names = val_dataset.classes

    # 텍스트 프롬프트 생성 및 인코딩
    print("\n1. Encoding text prompts...")
    text_prompts = get_text_prompts(class_names) # 기본 프롬프트 사용
    text_features = encode_text_prompts(model, text_prompts, device)
    print(f"  - Encoded {len(text_prompts)} classes, features shape: {text_features.shape}")

    # 이미지 특징 추출
    print("\n2. Extracting image features from validation set...")
    image_features, labels, _ = extract_features(model, val_loader, device)
    print(f"  - Extracted {len(image_features)} features, shape: {image_features.shape}")

    # Zero-Shot 예측
    print("\n3. Performing zero-shot prediction...")
    logits = 100.0 * torch.from_numpy(image_features).to(device) @ text_features.T
    probs = logits.softmax(dim=-1)
    preds = probs.argmax(dim=-1).cpu().numpy()
    print("  - Prediction complete.")

    # 평가
    print("\n4. Evaluating results...")
    evaluate(preds, labels, class_names, f"Zero-Shot_{args.model_name}", save_dir)
    
    print("\n" + "=" * 70)
    print("Zero-Shot Classification finished successfully!")
    print(f"All results saved in: {save_dir}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot Classification")
    
    # 데이터 및 모델 경로
    parser.add_argument("--val_dir", type=str, default="data/skin_dataset/Validation/01.원천데이터",
                        help="Validation data directory")
    parser.add_argument("--results_dir", type=str, default="results/clip_zero_shot",
                        help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="ViT-B/32",
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101'],
                        help="CLIP model to use")
                        
    # 데이터셋 및 로더 설정
    parser.add_argument("--classes", type=str, nargs='+', default=None,
                        help="List of classes to use (e.g., acne atopic). If None, uses all subdirectories.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 4),
                        help="Number of workers for DataLoader")

    args = parser.parse_args()
    main(args)