import numpy as np
import sys
import json
import os
import torch
from configs import TrainConfig
from dataloaders import make_data_loaders
from torch_trainer.models import build_model
from torch_trainer.trainer import Trainer
from utils.reproducibility import set_seed, set_reproducible_training, print_reproducibility_info
import wandb

import warnings
warnings.filterwarnings("ignore")


def main():

    config = TrainConfig.parse_arguments()

    # 실험 재현성 설정
    print("\n" + "="*70)
    print("Setting up reproducibility...")
    print("="*70)
    set_seed(config.random_state)
    worker_init_fn = set_reproducible_training()
    print_reproducibility_info()

    # 디바이스 정보 출력
    print(f"   PyTorch 버전: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA 사용 가능: O (GPU 모드로 학습)")
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   사용 가능한 GPU 수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"   CUDA 사용 가능: X (CPU 모드로 학습)")                    
    
    # config.save()

    # define data loader
    train_loader, valid_loader, test_loader = make_data_loaders(config)
    inputs, targets = next(iter(train_loader))
    
    # 클래스 이름 추출
    class_names = train_loader.dataset.get_class_names()
    print(f"Classes: {class_names}")
    
    # config의 모든 파라미터를 class_info.json에 저장하기 위해 딕셔너리로 변환
    class_info = vars(config)
    # 추가 정보 저장
    class_info['class_names'] = class_names
    class_info['n_class'] = len(class_names)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    # 초기 파라미터 저장
    with open(os.path.join(config.checkpoint_dir, 'class_info.json'), 'w') as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)

    # Define model & trainer
    model = build_model(model_name=config.model_name,
                        pre_trained=config.pre_trained,
                        n_class=config.n_class)

    trainer = Trainer(model=model, config=config)
    trainer.compile(ckpt_dir=config.checkpoint_dir,
                    loss_function=config.loss_function,
                    optimizer=config.optimizer,
                    scheduler=config.scheduler,
                    learnig_rate=config.lr_ae,
                    epochs=config.epochs,
                    local_rank=config.local_rank)

    best_loss = np.inf
    best_test_acc = 0.0
    epoch_start = 1
    results = {'train_loss': [],
               'valid_loss': [],
               'valid_acc': [],
               'test_acc': []}

    # Start Train & Evaluate
    for epoch in range(epoch_start, config.epochs+1):
        # train
        train_loss = trainer.train(epoch, train_loader)
        trainer.epoch_step()
        results['train_loss'].append(train_loss)

        valid_loss, valid_acc, early_stop = trainer.valid(epoch, valid_loader)
        results['valid_loss'].append(valid_loss)
        results['valid_acc'].append(valid_acc)

        # Append the last best test accuracy by default.
        # It will be updated if a new best model is found this epoch.
        results['test_acc'].append(best_test_acc)

        # test is performed only when validation loss improves
        if valid_loss < best_loss:
            best_loss = valid_loss
            
            # run test
            test_acc, cm = trainer.test(epoch, test_loader)
            best_test_acc = test_acc # Update the running best
            results['test_acc'][-1] = best_test_acc # Update this epoch's test accuracy

            # save results
            trainer.save(epoch, results, confusion_matrix=cm)
            
            # 최고 성능 모델의 정보도 업데이트
            class_info['best_epoch'] = epoch
            class_info['best_valid_loss'] = best_loss
            class_info['best_valid_acc'] = valid_acc
            class_info['best_test_acc'] = best_test_acc
            
            with open(os.path.join(config.checkpoint_dir, 'class_info.json'), 'w') as f:
                json.dump(class_info, f, indent=2)

        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    trainer.finish()

    print(f"\nTraining completed!")
    print(f"Best model saved at: {os.path.join(config.checkpoint_dir, 'model_last.pth')}")
    print(f"Class info saved at: {os.path.join(config.checkpoint_dir, 'class_info.json')}")
    print(f"To run inference, use:")
    print(f"python inference.py --model_path {os.path.join(config.checkpoint_dir, 'model_last.pth')} --model_name {config.model_name} --n_class {config.n_class} --class_names {' '.join(class_names)} --generate_gradcam")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()