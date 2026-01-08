import subprocess
import sys
import itertools
import copy

# =================================================================================
# 실험 템플릿 설정 (Grid Search 방식)
# =================================================================================

# --- 1. 하이퍼파라미터 그리드 ---
# 테스트하고 싶은 파라미터 값들을 리스트 형태로 정의합니다.
# 스크립트는 아래 리스트들의 모든 조합을 자동으로 생성하여 실험을 수행합니다.
MODELS = ["efficientnet_b0", "vit_tiny_patch16_224"]
AUGMENTATION_TYPES = ["base", "randaugment"]
MIXUP_OPTIONS = [True, False]
FOCAL_LOSS_OPTIONS = [True, False]
LEARNING_RATES = [0.001, 0.0005]


# --- 2. 모델별 특별 설정 ---
# 특정 모델에만 다르게 적용하고 싶은 파라미터가 있다면 여기에 정의합니다.
# 예를 들어, vit_b_16 모델은 크기가 크므로 작은 배치 사이즈를 적용합니다.
MODEL_SPECIFIC_PARAMS = {
    "efficientnet_b0": {"batch_size": 128},
    "vit_tiny_patch16_224": {"batch_size": 128},
    "vit_b_16": {"batch_size": 32}
}


# --- 3. 기본 파라미터 ---
# 모든 실험에 공통적으로 적용될 기본값입니다.
DEFAULT_PARAMS = {
    'data_name': 'skin',
    'epochs': 50,
    'wandb_project': 'skin_grid_search_V5'
}


def main():
    """
    정의된 하이퍼파라미터 그리드의 모든 조합에 대해 실험을 실행합니다.
    """
    # itertools.product를 사용하여 모든 경우의 수 조합을 생성
    param_grid = list(itertools.product(
        MODELS,
        AUGMENTATION_TYPES,
        MIXUP_OPTIONS,
        FOCAL_LOSS_OPTIONS,
        LEARNING_RATES
    ))
    total_experiments = len(param_grid)
    print(f"Total experiments to run: {total_experiments}")
    print("========================================")

    for i, (model_name, aug_type, use_mixup, use_focal_loss, lr) in enumerate(param_grid):
        
        # 1. 기본 파라미터 복사
        params = copy.deepcopy(DEFAULT_PARAMS)
        
        # 2. 현재 조합의 파라미터 추가
        params['model_name'] = model_name
        params['augmentation_type'] = aug_type
        params['use_mixup'] = use_mixup
        params['use_focal_loss'] = use_focal_loss
        params['lr_ae'] = lr
        
        # 3. 모델별 특별 파라미터 적용 (정의된 경우)
        if model_name in MODEL_SPECIFIC_PARAMS:
            params.update(MODEL_SPECIFIC_PARAMS[model_name])

        print(f"▶ Running experiment {i+1}/{total_experiments} with params:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        print("----------------------------------------")

        # 4. wandb를 위한 태그 자동 생성
        params['wandb_name_tags'] = list(params.keys())

        # 5. subprocess 실행을 위한 커맨드 생성
        command = [sys.executable, "main.py"]
        for key, value in params.items():
            # list 형태의 인자 처리 (예: wandb_name_tags)
            if isinstance(value, list):
                command.append(f'--{key}')
                command.extend(value)
            # boolean 값은 문자열로 변환
            elif isinstance(value, bool):
                command.append(f'--{key}')
                command.append(str(value))
            else:
                command.append(f'--{key}')
                command.append(str(value))

        try:
            subprocess.run(command, check=True)
            print(f"✔ Successfully completed experiment {i+1}/{total_experiments}")
            print("========================================")
        except subprocess.CalledProcessError as e:
            print(f"✖ Error running experiment {i+1}/{total_experiments} ✖")
            print(f"  Command: {' '.join(command)}")
            print(f"  Return code: {e.returncode}")
            sys.exit(1) # 오류 발생 시 스크립트 중단
        except FileNotFoundError:
            print("Error: main.py not found. Make sure you are in the correct directory.")
            sys.exit(1)

    print("🎉 All experiments completed successfully! 🎉")

if __name__ == "__main__":
    main()