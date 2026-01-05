import subprocess
import sys

# 실험 설정
MODELS = ["efficientnet_b0", "vit_b_16", "vit_tiny_patch16_224"]
AUGMENTATION_TYPES = ["base", "color", "mixed", "randaugment"]
LEARNING_RATE = 0.001
EPOCHS = 50
DATA_NAME = "skin"
WANDB_PROJECT = "skin_detection_experiments"

def main():
    """
    지정된 모델과 증강 유형에 대해 main.py를 실행하여 반복 실험을 수행합니다.
    """
    for model in MODELS:
        for aug_type in AUGMENTATION_TYPES:
            print(f"Running experiment with model: {model} and augmentation: {aug_type}")
            
            # wandb 실행 이름 설정 (예: model_efficientnet_b0_aug_base)
            wandb_run_name = f"model_{model}_aug_{aug_type}"

            command = [
                sys.executable,  # 현재 파이썬 인터프리터 사용
                "main.py",
                "--data_name", DATA_NAME,
                "--model_name", model,
                "--lr_ae", str(LEARNING_RATE),
                "--augmentation_type", aug_type,
                "--epochs", str(EPOCHS),
                "--wandb_project", WANDB_PROJECT,
                # wandb 실행 이름 태그를 동적으로 설정
                "--wandb_name_tags", "model_name", "augmentation_type"
            ]
            
            try:
                # main.py 실행
                subprocess.run(command, check=True)
                print(f"Successfully completed experiment with model: {model} and augmentation: {aug_type}")
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment with model: {model} and augmentation: {aug_type}")
                print(f"Command: {' '.join(command)}")
                print(f"Return code: {e.returncode}")
                # Stderr와 Stdout은 check=True일 때 자동으로 출력됩니다.
                # 오류 발생 시 다음 실험으로 넘어가지 않고 중단합니다.
                sys.exit(1)
            except FileNotFoundError:
                print(f"Error: main.py not found. Make sure you are in the correct directory.")
                sys.exit(1)

    print("All experiments completed.")

if __name__ == "__main__":
    main()
