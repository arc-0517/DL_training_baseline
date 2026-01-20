"CLIP Few-Shot Learning for Skin Disease Classification
- K-NN 또는 Linear Probe를 사용한 Few-Shot 학습 및 평가
"
import os
import torch
import torch.nn as nn
import clip
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import random

from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier

from clip_utils import (
    SkinDiseaseDataset,
    extract_features,
    evaluate,
)

# ============================================================ 
# Helper Functions
# ============================================================ 

def set_seed(seed):
    """재현성을 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def select_support_set(features, labels, n_shot, num_classes, seed):
    """각 클래스에서 N개의 샘플을 랜덤하게 선택하여 Support Set 구성"""
    set_seed(seed)
    support_features = []
    support_labels = []

    for i in range(num_classes):
        # 현재 클래스에 해당하는 샘플 필터링
        class_indices = np.where(labels == i)[0]
        
        # n_shot보다 샘플 수가 적으면 가능한 만큼만 사용
        num_samples = min(n_shot, len(class_indices))
        
        # 랜덤하게 샘플 선택
        support_indices = np.random.choice(class_indices, num_samples, replace=False)
        
        support_features.append(features[support_indices])
        support_labels.append(labels[support_indices])
        
    return np.concatenate(support_features), np.concatenate(support_labels)

# ============================================================ 
# K-NN Classifier
# ============================================================ 

def run_knn(support_features, support_labels, query_features, k):
    """K-NN 분류기를 학습하고 예측"""
    print(f"\nRunning K-NN with k={k}...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(support_features, support_labels)
    preds = knn.predict(query_features)
    return preds

# ============================================================ 
# Linear Probe (MLP Classifier)
# ============================================================ 

class LinearProbe(nn.Module):
    """간단한 MLP 분류기"""
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

def train_linear_probe(probe, train_loader, val_loader, args, device):
    """Linear Probe 모델 학습"""
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nTraining Linear Probe for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = probe(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        probe.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = probe(batch_features)
                val_loss += criterion(outputs, batch_labels).item()
        
        val_loss /= len(val_loader)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(probe.state_dict(), 'best_probe.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    probe.load_state_dict(torch.load('best_probe.pth'))
    return probe

def predict_with_probe(probe, features, device, batch_size=64):
    """학습된 Linear Probe로 예측"""
    probe.eval()
    all_preds = []
    feature_tensor = torch.from_numpy(features).float()
    dataset = TensorDataset(feature_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting with Linear Probe"):
            batch_features = batch[0].to(device)
            outputs = probe(batch_features)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


# ============================================================ 
# Main Function
# ============================================================ 

def main(args):
    """메인 실행 함수"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 결과 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_info = f"{args.method}_{args.n_shot}shot"
    if args.method == 'knn':
        method_info += f"_k{args.k}"
    
    save_dir = Path(args.results_dir) / f"{args.model_name.replace('/', '-')}_{method_info}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLIP Few-Shot Classification")
    print(f"  - Method: {args.method.upper()}")
    print(f"  - N-shot: {args.n_shot}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Train Dir: {args.train_dir}")
    print(f"  - Val Dir: {args.val_dir}")
    print(f"  - Results will be saved to: {save_dir}")
    print("=" * 70)

    # 1. CLIP 모델 로드 및 특징 추출
    model, preprocess = clip.load(args.model_name, device=device)
    model.eval()

    train_dataset = SkinDiseaseDataset(args.train_dir, preprocess, args.classes)
    val_dataset = SkinDiseaseDataset(args.val_dir, preprocess, args.classes)
    
    # 클래스 정보 일치 확인
    assert train_dataset.classes == val_dataset.classes, "Train and validation classes do not match!"
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # 데이터로더
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers)

    print("\n1. Extracting features from training and validation sets...")
    train_features, train_labels, _ = extract_features(model, train_loader, device)
    val_features, val_labels, _ = extract_features(model, val_loader, device)
    print(f"  - Extracted {len(train_features)} train features.")
    print(f"  - Extracted {len(val_features)} validation features.")

    # 2. Support Set 선택
    print(f"\n2. Selecting {args.n_shot}-shot support set...")
    support_features, support_labels = select_support_set(train_features, train_labels, args.n_shot, num_classes, args.seed)
    print(f"  - Support set size: {len(support_features)}")
    
    # 3. 선택된 방법 실행 및 평가
    if args.method == 'knn':
        preds = run_knn(support_features, support_labels, val_features, args.k)
        method_name = f"KNN {args.n_shot}-shot (k={args.k})"
        evaluate(preds, val_labels, class_names, method_name, save_dir)

    elif args.method == 'linear_probe':
        input_dim = support_features.shape[1]
        probe = LinearProbe(input_dim, num_classes, args.hidden_dim, args.dropout).to(device)

        # Support set으로 학습 데이터로더 생성
        support_dataset = TensorDataset(torch.from_numpy(support_features).float(), torch.from_numpy(support_labels).long())
        support_loader = DataLoader(support_dataset, batch_size=min(args.batch_size, len(support_dataset)), shuffle=True)
        
        # 전체 Val set으로 검증 데이터로더 생성
        val_feature_dataset = TensorDataset(torch.from_numpy(val_features).float(), torch.from_numpy(val_labels).long())
        val_feature_loader = DataLoader(val_feature_dataset, batch_size=args.batch_size)

        probe = train_linear_probe(probe, support_loader, val_feature_loader, args, device)
        preds = predict_with_probe(probe, val_features, device, args.batch_size)
        
        method_name = f"Linear Probe {args.n_shot}-shot"
        evaluate(preds, val_labels, class_names, method_name, save_dir)
    
    if os.path.exists('best_probe.pth'):
        os.remove('best_probe.pth')

    print("\n" + "=" * 70)
    print("Few-Shot Classification finished successfully!")
    print(f"All results saved in: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Few-Shot Classification")

    # 공통 인자
    parser.add_argument("--method", type=str, default="linear_probe", choices=['knn', 'linear_probe'])
    parser.add_argument("--train_dir", type=str, default="data/skin_dataset/Training/01.원천데이터")
    parser.add_argument("--val_dir", type=str, default="data/skin_dataset/Validation/01.원천데이터")
    parser.add_argument("--results_dir", type=str, default="results/clip_few_shot")
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--classes", type=str, nargs='+', default=None)
    parser.add_argument("--n_shot", type=int, default=10, help="Number of samples per class for the support set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=min(os.cpu_count(), 4))
    parser.add_argument("--seed", type=int, default=42)

    # K-NN 인자
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors for K-NN")

    # Linear Probe 인자
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for linear probe")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    args = parser.parse_args()
    main(args)