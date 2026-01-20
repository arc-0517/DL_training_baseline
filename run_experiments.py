import subprocess
import sys
import itertools

# 1. 모델 설정 (EfficientNet만 사용)
EFFICIENTNET_MODELS = ['efficientnet_b1', 'efficientnet_b2']  # b1, b2만 실험

# 2. 배치 사이즈 고정
BATCH_SIZE = 64

# 3. Augmentation 설정
# AUGMENTATION_TYPES = ['base', 'mixed']
AUGMENTATION_TYPES = ['base', 'color', 'mixed', 'randaugment']

# 4. Mixup 적용 여부
MIXUP_OPTIONS = [True, False]

# 5. Loss 함수 선택 (Focal Loss 사용 여부)
FOCAL_LOSS_OPTIONS = [True, False]

# 6. 기타 설정
LEARNING_RATE = 0.001
EPOCHS = 50
DATA_NAME = "skin"
RANDOM_STATE = 42  # 재현성을 위한 seed

# 7. 선택된 레이블 (None이면 전체 레이블 사용)
# 전체 레이블: ['seborrheic', 'rosacea', 'normal', 'acne', 'atopic', 'psoriasis']
SELECTED_LABELS = ['normal', 'acne', 'rosacea', 'atopic']  # seborrheic, psoriasis 제외

# --- 스크립트 본체 ---

def main():
    """
    지정된 설정의 모든 조합에 대해 main.py를 실행하여 반복 실험을 수행합니다.
    """
    # 모델 리스트 및 프로젝트 이름 설정
    models = EFFICIENTNET_MODELS
    wandb_project = "efficientnet_skin_labelsubset_experiments"

    # itertools.product를 사용하여 모든 경우의 수 조합을 생성 (모델 포함)
    experiment_cases = list(itertools.product(models, AUGMENTATION_TYPES, MIXUP_OPTIONS, FOCAL_LOSS_OPTIONS))
    total_experiments = len(experiment_cases)
    print(f"Model Family: EfficientNet")
    print(f"Models: {models}")
    print(f"WandB Project: {wandb_project}")
    print(f"Selected Labels: {SELECTED_LABELS if SELECTED_LABELS else 'All labels'}")
    print(f"Total experiments to run: {total_experiments}")
    print("---")

    for i, (model, aug_type, use_mixup, use_focal_loss) in enumerate(experiment_cases):

        print(f"Running experiment {i+1}/{total_experiments}:")
        print(f"  - Model: {model}")
        print(f"  - Augmentation: {aug_type}")
        print(f"  - Use Mixup: {use_mixup}")
        print(f"  - Use Focal Loss: {use_focal_loss}")
        print(f"  - Batch Size: {BATCH_SIZE}")
        print(f"  - Selected Labels: {SELECTED_LABELS if SELECTED_LABELS else 'All labels'}")
        print("---")

        command = [
            sys.executable,
            "main.py",
            "--data_name", DATA_NAME,
            "--model_name", model,
            "--lr_ae", str(LEARNING_RATE),
            "--epochs", str(EPOCHS),
            "--wandb_project", wandb_project,
            "--batch_size", str(BATCH_SIZE),
            "--augmentation_type", aug_type,
            "--use_mixup", str(use_mixup),
            "--use_focal_loss", str(use_focal_loss),
            "--random_state", str(RANDOM_STATE),
            "--wandb_name_tags", "model_name", "augmentation_type", "use_mixup", "use_focal_loss"
        ]

        # selected_labels 추가 (설정된 경우에만)
        if SELECTED_LABELS:
            command.extend(["--selected_labels"] + SELECTED_LABELS)

        try:
            subprocess.run(command, check=True)
            print(f"--- Successfully completed experiment {i+1}/{total_experiments} ---")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running experiment {i+1}/{total_experiments} !!!")
            print(f"Command: {' '.join(command)}")
            print(f"Return code: {e.returncode}")
            sys.exit(1) # 오류 발생 시 스크립트 중단
        except FileNotFoundError:
            print(f"Error: main.py not found. Make sure you are in the correct directory.")
            sys.exit(1)

    print("All experiments completed.")

if __name__ == "__main__":
    main()
