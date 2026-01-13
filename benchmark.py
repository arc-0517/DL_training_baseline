"""
모델 성능 벤치마크 스크립트

PyTorch와 ONNX 모델의 추론 속도와 메모리 사용량을 비교합니다.
모델 경량화의 효과를 정량적으로 측정할 수 있습니다.

사용 예시:
    python benchmark.py \
        --pytorch_model_path save_results/dataset+skin/model+efficientnet_b0/2024-01-01_12-00-00 \
        --onnx_model_dir onnx_models
"""

import torch
import os
import argparse
import json
import time
import psutil
import numpy as np
import pandas as pd

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    from transformers import SegformerFeatureExtractor as SegformerImageProcessor, SegformerForSemanticSegmentation

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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


def measure_inference(model, model_type, input_data, device, num_iterations=100):
    """
    모델의 추론 시간과 메모리 사용량 측정

    Args:
        model: 벤치마크할 모델 (PyTorch 모듈 또는 ONNX 세션)
        model_type (str): 'pytorch' 또는 'onnx'
        input_data: 입력 데이터 (PyTorch의 경우 텐서, ONNX의 경우 딕셔너리)
        device: 사용할 디바이스 (cuda/cpu)
        num_iterations (int): 반복 횟수 (기본값: 100)

    Returns:
        dict: 성능 지표 (memory_mb, avg_time_ms, std_time_ms, min_time_ms, max_time_ms)
    """
    # 메모리 사용량 측정
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / (1024 * 1024)

        # 한 번 실행하여 피크 메모리 측정
        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(input_data)
        elif model_type == 'onnx':
            _ = model.run(None, input_data)

        mem_after = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        memory_mb = mem_after - mem_before
        torch.cuda.empty_cache()
    else:  # CPU
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(input_data)
        elif model_type == 'onnx':
            _ = model.run(None, input_data)

        mem_after = process.memory_info().rss / (1024 * 1024)
        memory_mb = mem_after - mem_before

    # 추론 속도 측정
    times = []

    # 워밍업 (캐시 예열)
    for _ in range(10):
        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(input_data)
        elif model_type == 'onnx':
            _ = model.run(None, input_data)

    # 실제 측정
    for _ in range(num_iterations):
        start_time = time.perf_counter()

        if model_type == 'pytorch':
            with torch.no_grad():
                _ = model(input_data)
        elif model_type == 'onnx':
            _ = model.run(None, input_data)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # ms로 변환

    return {
        "memory_mb": round(memory_mb, 2),
        "avg_time_ms": round(np.mean(times), 4),
        "std_time_ms": round(np.std(times), 4),
        "min_time_ms": round(np.min(times), 4),
        "max_time_ms": round(np.max(times), 4)
    }


def load_pytorch_models(args, device):
    """
    PyTorch 모델 로드

    Args:
        args: 명령줄 인자
        device: 사용할 디바이스

    Returns:
        tuple: (분류 모델, 세그멘테이션 모델, 세그멘테이션 프로세서, 설정)
    """
    print("\n[1/4] PyTorch 모델 로딩 중...")

    # 설정 파일 로드
    clf_config_path = os.path.join(args.pytorch_model_path, 'class_info.json')
    with open(clf_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 분류 모델 로드
    print("  분류 모델 (EfficientNet) 로딩 중...")
    clf_model_pt = build_model(
        model_name=config['model_name'],
        pre_trained=False,
        n_class=config['n_class']
    )
    clf_weight_path = os.path.join(args.pytorch_model_path, 'model_last.pth')
    checkpoint = torch.load(clf_weight_path, map_location=device)
    clf_model_pt.load_state_dict(checkpoint['state_dict'])
    clf_model_pt.eval()
    clf_model_pt.to(device)

    seg_model_pt = None
    seg_processor = None

    # 세그멘테이션 모델 로드 (선택적)
    if not args.skip_segmentation:
        try:
            print("  세그멘테이션 모델 (SegFormer) 로딩 중...")
            seg_model_pt = SegformerForSemanticSegmentation.from_pretrained(
                "jonathandinu/face-parsing"
            )
            seg_processor = SegformerImageProcessor.from_pretrained(
                "jonathandinu/face-parsing"
            )
            seg_model_pt.eval()
            seg_model_pt.to(device)
        except Exception as e:
            print(f"  ⚠ 세그멘테이션 모델 로드 실패: {e}")
            seg_model_pt = None
            seg_processor = None

    return clf_model_pt, seg_model_pt, seg_processor, config


def benchmark_pytorch_models(clf_model_pt, seg_model_pt, seg_processor, config, device):
    """
    PyTorch 모델 벤치마크

    Args:
        clf_model_pt: 분류 모델
        seg_model_pt: 세그멘테이션 모델
        seg_processor: 세그멘테이션 프로세서
        config (dict): 모델 설정
        device: 사용할 디바이스

    Returns:
        list: 벤치마크 결과 리스트
    """
    print("\n[2/4] PyTorch 모델 벤치마크 실행 중...")

    results = []

    # 입력 데이터 준비
    clf_input_pt = torch.randn(1, 3, config['img_size'], config['img_size'], device=device)

    # 분류 모델 벤치마크
    clf_pt_size = get_pytorch_model_size_mb(clf_model_pt)
    clf_perf_pt = measure_inference(clf_model_pt, 'pytorch', clf_input_pt, device)
    results.append({
        "모델": "분류 모델",
        "프레임워크": "PyTorch",
        "크기 (MB)": clf_pt_size,
        "평균 추론 시간 (ms)": clf_perf_pt['avg_time_ms'],
        "표준편차 (ms)": clf_perf_pt['std_time_ms']
    })
    print("  분류 모델 벤치마크 완료")

    # 세그멘테이션 모델 벤치마크
    if seg_model_pt is not None and seg_processor is not None:
        seg_input_pt = seg_processor(
            images=[np.zeros((512, 512, 3), dtype=np.uint8)],
            return_tensors="pt"
        )['pixel_values'].to(device)

        seg_pt_size = get_pytorch_model_size_mb(seg_model_pt)
        seg_perf_pt = measure_inference(seg_model_pt, 'pytorch', seg_input_pt, device)
        results.append({
            "모델": "세그멘테이션 모델",
            "프레임워크": "PyTorch",
            "크기 (MB)": seg_pt_size,
            "평균 추론 시간 (ms)": seg_perf_pt['avg_time_ms'],
            "표준편차 (ms)": seg_perf_pt['std_time_ms']
        })
        print("  세그멘테이션 모델 벤치마크 완료")

    return results


def benchmark_onnx_models(args, config, device, results):
    """
    ONNX 모델 벤치마크

    Args:
        args: 명령줄 인자
        config (dict): 모델 설정
        device: 사용할 디바이스
        results (list): 벤치마크 결과 리스트

    Returns:
        list: 업데이트된 벤치마크 결과 리스트
    """
    print("\n[3/4] ONNX 모델 벤치마크 실행 중...")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']

    onnx_models = {
        "분류 모델": "clf_model.onnx",
        "세그멘테이션 모델": "seg_model.onnx"
    }

    # 입력 데이터 준비
    clf_input_np = np.random.randn(1, 3, config['img_size'], config['img_size']).astype(np.float32)

    for model_name, filename in onnx_models.items():
        model_path = os.path.join(args.onnx_model_dir, filename)

        if not os.path.exists(model_path):
            print(f"  ⚠ {filename} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        try:
            session = ort.InferenceSession(model_path, providers=providers)

            # 입력 데이터 준비
            if model_name == "분류 모델":
                input_name = session.get_inputs()[0].name
                onnx_input = {input_name: clf_input_np}
            else:  # 세그멘테이션 모델
                seg_input_np = np.zeros((1, 3, 512, 512), dtype=np.float32)
                input_name = session.get_inputs()[0].name
                onnx_input = {input_name: seg_input_np}

            onnx_size = get_model_size_mb(model_path)
            perf_onnx = measure_inference(session, 'onnx', onnx_input, device)

            results.append({
                "모델": model_name,
                "프레임워크": "ONNX",
                "크기 (MB)": onnx_size,
                "평균 추론 시간 (ms)": perf_onnx['avg_time_ms'],
                "표준편차 (ms)": perf_onnx['std_time_ms']
            })
            print(f"  {model_name} 벤치마크 완료")

        except Exception as e:
            print(f"  ⚠ {filename} 로드 실패: {e}")
            continue

    return results


def display_results(results):
    """
    벤치마크 결과 출력

    Args:
        results (list): 벤치마크 결과 리스트
    """
    print("\n[4/4] 결과 분석")
    print("="*70)

    if not results:
        print("벤치마크할 모델이 없습니다.")
        return

    df = pd.DataFrame(results)

    # 속도 향상 및 크기 감소율 계산
    df['속도 향상'] = ''
    df['크기 감소율'] = ''

    for model_name in df['모델'].unique():
        pt_row = df[(df['모델'] == model_name) & (df['프레임워크'] == 'PyTorch')]
        onnx_row = df[(df['모델'] == model_name) & (df['프레임워크'] == 'ONNX')]

        if not pt_row.empty and not onnx_row.empty:
            pt_idx = pt_row.index[0]
            onnx_idx = onnx_row.index[0]

            pt_time = pt_row['평균 추론 시간 (ms)'].values[0]
            onnx_time = onnx_row['평균 추론 시간 (ms)'].values[0]
            speedup = pt_time / onnx_time

            pt_size = pt_row['크기 (MB)'].values[0]
            onnx_size = onnx_row['크기 (MB)'].values[0]
            size_reduction = ((pt_size - onnx_size) / pt_size) * 100

            df.at[onnx_idx, '속도 향상'] = f"{speedup:.2f}x"
            df.at[onnx_idx, '크기 감소율'] = f"{size_reduction:.1f}%"

    # 표 출력
    print("\n성능 비교:")
    print(df.to_string(index=False))

    # 요약 출력
    print("\n" + "="*70)
    print("요약")
    print("="*70)

    for model_name in df['모델'].unique():
        pt_row = df[(df['모델'] == model_name) & (df['프레임워크'] == 'PyTorch')]
        onnx_row = df[(df['모델'] == model_name) & (df['프레임워크'] == 'ONNX')]

        if not pt_row.empty and not onnx_row.empty:
            pt_size = pt_row['크기 (MB)'].values[0]
            onnx_size = onnx_row['크기 (MB)'].values[0]
            pt_time = pt_row['평균 추론 시간 (ms)'].values[0]
            onnx_time = onnx_row['평균 추론 시간 (ms)'].values[0]

            size_reduction = ((pt_size - onnx_size) / pt_size) * 100
            speedup = pt_time / onnx_time

            print(f"\n{model_name}:")
            print(f"  모델 크기:   {pt_size:.2f} MB → {onnx_size:.2f} MB ({size_reduction:.1f}% 감소)")
            print(f"  추론 시간:   {pt_time:.4f} ms → {onnx_time:.4f} ms ({speedup:.2f}x 향상)")

    print("\n" + "="*70)


def main(args):
    """
    메인 함수 - PyTorch와 ONNX 모델의 성능 비교

    Args:
        args: 명령줄 인자
    """
    if ort is None:
        raise ImportError("onnxruntime이 설치되지 않았습니다. 'pip install onnxruntime'로 설치하세요.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("모델 성능 벤치마크: PyTorch vs ONNX")
    print("="*70)
    print(f"디바이스: {device}")
    print(f"반복 횟수: 100회")
    print("="*70)

    # PyTorch 모델 로드
    clf_model_pt, seg_model_pt, seg_processor, config = load_pytorch_models(args, device)

    # PyTorch 모델 벤치마크
    results = benchmark_pytorch_models(clf_model_pt, seg_model_pt, seg_processor, config, device)

    # ONNX 모델 벤치마크
    results = benchmark_onnx_models(args, config, device, results)

    # 결과 출력
    display_results(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch와 ONNX 모델의 추론 성능을 비교합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용법
  python benchmark.py --pytorch_model_path save_results/.../2024-01-01_12-00-00

  # ONNX 모델 디렉토리 지정
  python benchmark.py --pytorch_model_path save_results/.../2024-01-01_12-00-00 --onnx_model_dir my_onnx_models

  # 세그멘테이션 모델 제외
  python benchmark.py --pytorch_model_path save_results/.../2024-01-01_12-00-00 --skip_segmentation
        """
    )

    parser.add_argument(
        '--pytorch_model_path',
        type=str,
        required=True,
        help='학습된 PyTorch 모델이 저장된 디렉토리 경로'
    )

    parser.add_argument(
        '--onnx_model_dir',
        type=str,
        default='onnx_models',
        help='ONNX 모델이 저장된 디렉토리 (기본값: onnx_models)'
    )

    parser.add_argument(
        '--skip_segmentation',
        action='store_true',
        help='세그멘테이션 모델 벤치마크를 건너뜁니다'
    )

    args = parser.parse_args()
    main(args)
