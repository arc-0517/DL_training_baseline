"""
실험 재현성을 보장하기 위한 유틸리티 함수들

PyTorch, NumPy, Python random 모듈의 랜덤 시드를 고정하고
결정론적(deterministic) 동작을 보장합니다.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=0):
    """
    모든 랜덤 시드를 고정하여 실험 재현성을 보장합니다.

    Args:
        seed (int): 랜덤 시드 값 (기본값: 0)
    """
    # Python random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시

    # CUDA 결정론적 동작 설정
    # CuDNN 결정론적 모드 활성화 (성능은 약간 저하될 수 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 추가 환경 변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA 10.2 이상

    print(f"[OK] Random seed set to {seed}")
    print(f"[OK] Deterministic mode enabled")


def set_reproducible_training():
    """
    PyTorch 학습의 재현성을 최대한 보장하도록 설정합니다.

    주의: 일부 연산(예: atomic operations)은 여전히 비결정적일 수 있습니다.
    """
    # PyTorch 결정론적 알고리즘 사용 (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(True)
        print("[OK] Using deterministic algorithms")
    except AttributeError:
        # PyTorch 버전이 낮은 경우
        print("[WARNING] torch.use_deterministic_algorithms not available (requires PyTorch 1.8+)")
        print("  Using cudnn.deterministic instead")

    # 워커 프로세스의 시드도 고정
    def worker_init_fn(worker_id):
        """DataLoader 워커의 랜덤 시드 고정"""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn


def print_reproducibility_info():
    """재현성 관련 설정 정보를 출력합니다."""
    print("\n" + "="*70)
    print("Reproducibility Settings")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print(f"CuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"CuDNN benchmark: {torch.backends.cudnn.benchmark}")

    try:
        print(f"Deterministic algorithms: {torch.are_deterministic_algorithms_enabled()}")
    except AttributeError:
        print(f"Deterministic algorithms: Not available (PyTorch < 1.8)")

    print("="*70 + "\n")


def get_worker_init_fn(seed=0):
    """
    DataLoader에서 사용할 worker_init_fn을 반환합니다.

    Args:
        seed (int): 베이스 시드 값

    Returns:
        function: worker_init_fn
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return worker_init_fn
