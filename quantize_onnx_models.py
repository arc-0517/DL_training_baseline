"""
ONNX 모델 변환 스크립트

PyTorch로 학습된 모델을 ONNX 형식으로 변환하여 추론 속도를 최적화합니다.
ONNX 형식은 다양한 플랫폼에서 실행 가능하며, 추론 성능이 크게 향상됩니다.

사용 예시:
    python quantize_onnx_models.py \
        --model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
        --output_dir onnx_models
"""

import torch
import os
import argparse
import json
import numpy as np

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    from transformers import SegformerFeatureExtractor as SegformerImageProcessor, SegformerForSemanticSegmentation

import onnx
from torch_trainer.models import build_model


def get_model_size_mb(path):
    """
    파일 크기를 MB 단위로 반환

    Args:
        path (str): 모델 파일 경로

    Returns:
        float: 파일 크기 (MB 단위)
    """
    if not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / (1024 * 1024), 2)


def get_pytorch_model_size_mb(model):
    """
    PyTorch 모델의 파라미터 크기를 MB 단위로 계산

    Args:
        model: PyTorch 모델

    Returns:
        float: 모델 크기 (MB 단위)
    """
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(size_bytes / (1024 * 1024), 2)


def convert_classification_model(args, device, results):
    """
    분류 모델(EfficientNet)을 ONNX로 변환

    Args:
        args: 명령줄 인자
        device: 사용할 디바이스 (cuda/cpu)
        results (dict): 결과 저장용 딕셔너리
    """
    print("\n" + "="*70)
    print("[1/2] 분류 모델 변환 (Classification Model - EfficientNet)")
    print("="*70)

    # 설정 파일 로드
    config_path = os.path.join(args.model_path, 'class_info.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_name = config['model_name']
    n_class = config['n_class']
    img_size = config['img_size']

    print(f"모델: {model_name}")
    print(f"클래스 수: {n_class}")
    print(f"입력 크기: {img_size}x{img_size}")

    # PyTorch 모델 로드
    print("\nPyTorch 모델 로딩 중...")
    pytorch_model = build_model(
        model_name=model_name,
        pre_trained=False,
        n_class=n_class
    )

    model_weight_path = os.path.join(args.model_path, 'model_last.pth')
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"모델 가중치를 찾을 수 없습니다: {model_weight_path}")

    checkpoint = torch.load(model_weight_path, map_location=device)
    pytorch_model.load_state_dict(checkpoint['state_dict'])
    pytorch_model.eval()
    pytorch_model.to(device)

    pytorch_size = get_pytorch_model_size_mb(pytorch_model)
    print(f"PyTorch 모델 크기: {pytorch_size} MB")

    # ONNX로 변환
    print("\nONNX 형식으로 변환 중...")
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    onnx_path = os.path.join(args.output_dir, 'clf_model.onnx')

    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,  # 상수 폴딩으로 그래프 최적화
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    onnx_size = get_model_size_mb(onnx_path)
    print(f"ONNX 모델 저장됨: {onnx_path}")
    print(f"ONNX 모델 크기: {onnx_size} MB")

    # ONNX 모델 검증
    onnx.checker.check_model(onnx_path)
    print("✓ ONNX 모델 검증 완료")

    results['분류 모델'] = {
        'pytorch_size': pytorch_size,
        'onnx_size': onnx_size
    }


def convert_segmentation_model(args, device, results):
    """
    세그멘테이션 모델(SegFormer)을 ONNX로 변환

    Args:
        args: 명령줄 인자
        device: 사용할 디바이스 (cuda/cpu)
        results (dict): 결과 저장용 딕셔너리
    """
    print("\n" + "="*70)
    print("[2/2] 세그멘테이션 모델 변환 (Segmentation Model - SegFormer)")
    print("="*70)

    print("사전 학습된 SegFormer 모델 로딩 중...")
    print("(인터넷 연결 필요 - 처음 실행 시 모델을 다운로드합니다)")

    try:
        pytorch_model = SegformerForSemanticSegmentation.from_pretrained(
            "jonathandinu/face-parsing"
        )
        seg_processor = SegformerImageProcessor.from_pretrained(
            "jonathandinu/face-parsing"
        )
        pytorch_model.eval()
        pytorch_model.to(device)

        pytorch_size = get_pytorch_model_size_mb(pytorch_model)
        print(f"PyTorch 모델 크기: {pytorch_size} MB")

        # ONNX로 변환
        print("\nONNX 형식으로 변환 중...")
        dummy_input = seg_processor(
            images=[np.zeros((512, 512, 3), dtype=np.uint8)],
            return_tensors="pt"
        )['pixel_values'].to(device)

        onnx_path = os.path.join(args.output_dir, 'seg_model.onnx')

        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,  # 상수 폴딩으로 그래프 최적화
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )

        onnx_size = get_model_size_mb(onnx_path)
        print(f"ONNX 모델 저장됨: {onnx_path}")
        print(f"ONNX 모델 크기: {onnx_size} MB")

        # ONNX 모델 검증
        onnx.checker.check_model(onnx_path)
        print("✓ ONNX 모델 검증 완료")

        results['세그멘테이션 모델'] = {
            'pytorch_size': pytorch_size,
            'onnx_size': onnx_size
        }

    except Exception as e:
        print(f"\n⚠ 세그멘테이션 모델 변환 실패: {e}")
        print("인터넷 연결을 확인하거나, 이전에 변환된 모델을 사용하세요.")
        results['세그멘테이션 모델'] = None


def print_summary(results, output_dir):
    """
    변환 결과 요약 출력

    Args:
        results (dict): 변환 결과 딕셔너리
        output_dir (str): 출력 디렉토리 경로
    """
    print("\n" + "="*70)
    print("모델 변환 요약")
    print("="*70)

    for model_name, sizes in results.items():
        if sizes is None:
            print(f"\n{model_name}: 변환 실패")
            continue

        pytorch_size = sizes['pytorch_size']
        onnx_size = sizes['onnx_size']
        reduction = ((pytorch_size - onnx_size) / pytorch_size * 100) if pytorch_size > 0 else 0

        print(f"\n{model_name}:")
        print(f"  PyTorch: {pytorch_size:>8.2f} MB")
        print(f"  ONNX:    {onnx_size:>8.2f} MB")
        print(f"  감소율:  {reduction:>6.1f}%")

    print("\n" + "="*70)
    print(f"변환된 모델 저장 위치: {os.path.abspath(output_dir)}")
    print("="*70)
    print("\n다음 단계: benchmark.py를 실행하여 성능을 비교하세요.")


def main(args):
    """
    메인 함수 - PyTorch 모델을 ONNX로 변환

    Args:
        args: 명령줄 인자
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("PyTorch → ONNX 모델 변환")
    print("="*70)
    print(f"디바이스: {device}")
    print(f"ONNX opset 버전: 14")

    os.makedirs(args.output_dir, exist_ok=True)

    # 결과 저장용 딕셔너리
    results = {}

    # 1. 분류 모델 변환
    convert_classification_model(args, device, results)

    # 2. 세그멘테이션 모델 변환 (선택적)
    if not args.skip_segmentation:
        convert_segmentation_model(args, device, results)

    # 3. 결과 요약 출력
    print_summary(results, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch 모델을 ONNX 형식으로 변환하여 추론 속도를 최적화합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용법
  python quantize_onnx_models.py --model_path save_results/.../2024-01-01_12-00-00

  # 출력 디렉토리 지정
  python quantize_onnx_models.py --model_path save_results/.../2024-01-01_12-00-00 --output_dir my_onnx_models

  # 세그멘테이션 모델 제외
  python quantize_onnx_models.py --model_path save_results/.../2024-01-01_12-00-00 --skip_segmentation
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='학습된 PyTorch 모델이 저장된 디렉토리 경로 (class_info.json과 model_last.pth 포함)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='onnx_models',
        help='ONNX 모델을 저장할 디렉토리 (기본값: onnx_models)'
    )

    parser.add_argument(
        '--skip_segmentation',
        action='store_true',
        help='세그멘테이션 모델 변환을 건너뜁니다 (인터넷 연결이 없을 때 유용)'
    )

    args = parser.parse_args()
    main(args)
