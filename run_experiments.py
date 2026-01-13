import subprocess
import sys
import itertools

# 1. 모델 설정
MODEL = "efficientnet_b0"

# 2. 배치 사이즈 고정
BATCH_SIZE = 64

# 3. Augmentation 설정
# AUGMENTATION_TYPES = ['base', 'mixed']
AUGMENTATION_TYPES = ['base', 'color', 'randaugment']

# 4. Mixup 적용 여부
MIXUP_OPTIONS = [True, False]

# 5. Loss 함수 선택 (Focal Loss 사용 여부)
FOCAL_LOSS_OPTIONS = [True, False]

# 6. 기타 설정
LEARNING_RATE = 0.001
EPOCHS = 2
DATA_NAME = "skin"
WANDB_PROJECT = "skin_exp_efficientnet_b0_test"

# --- 스크립트 본체 ---

def main():
    """
    지정된 설정의 모든 조합에 대해 main.py를 실행하여 반복 실험을 수행합니다.
    """
    # itertools.product를 사용하여 모든 경우의 수 조합을 생성
    experiment_cases = list(itertools.product(AUGMENTATION_TYPES, MIXUP_OPTIONS, FOCAL_LOSS_OPTIONS))
    total_experiments = len(experiment_cases)
    print(f"Total experiments to run: {total_experiments}")
    print("---")

    for i, (aug_type, use_mixup, use_focal_loss) in enumerate(experiment_cases):
        
        print(f"Running experiment {i+1}/{total_experiments}:")
        print(f"  - Model: {MODEL}")
        print(f"  - Augmentation: {aug_type}")
        print(f"  - Use Mixup: {use_mixup}")
        print(f"  - Use Focal Loss: {use_focal_loss}")
        print(f"  - Batch Size: {BATCH_SIZE}")
        print("---")

        command = [
            sys.executable,
            "main.py",
            "--data_name", DATA_NAME,
            "--model_name", MODEL,
            "--lr_ae", str(LEARNING_RATE),
            "--epochs", str(EPOCHS),
            "--wandb_project", WANDB_PROJECT,
            "--batch_size", str(BATCH_SIZE),
            "--augmentation_type", aug_type,
            "--use_mixup", str(use_mixup),
            "--use_focal_loss", str(use_focal_loss),
            "--wandb_name_tags", "model_name", "augmentation_type", "use_mixup", "use_focal_loss"
        ]
        
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
