import subprocess
import sys

# 모델 리스트
MODELS = ["resnet50", "efficientnet_b0", "efficientnet_b1"]
# 옵티마이저 리스트
OPTIMIZERS = ["adam", "sgd", "adamW"]

# wandb 프로젝트 이름
PROJECT_NAME = "dogcat_test"

def main():
    """
    각 모델과 옵티마이저에 대해 main.py를 실행하여 반복 실험을 수행합니다.
    """
    for model in MODELS:
        for optimizer in OPTIMIZERS:
            print(f"Running experiment with model: {model} and optimizer: {optimizer}")
            command = [
                sys.executable,  # 현재 파이썬 인터프리터 사용
                "main.py",
                "--wandb_project",
                PROJECT_NAME,
                "--model_name",
                model,
                "--optimizer",
                optimizer,
            ]
            
            try:
                # main.py 실행
                subprocess.run(command, check=True)
                print(f"Successfully completed experiment with model: {model} and optimizer: {optimizer}")
            except subprocess.CalledProcessError as e:
                print(f"Error running experiment with model: {model} and optimizer: {optimizer}")
                print(f"Command: {' '.join(command)}")
                print(f"Return code: {e.returncode}")
                print(f"Output:\n{e.output}")
                break  # 오류 발생 시 중단
            except FileNotFoundError:
                print(f"Error: main.py not found. Make sure you are in the correct directory.")
                break
        else:
            continue
        break

    print("All experiments completed.")

if __name__ == "__main__":
    main()