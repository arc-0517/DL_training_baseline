import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

class Trainer(object):
    def __init__(self,
                 model: nn.Module):
        super(Trainer, self).__init__()

        self.model = model
        self.compiled = False

    def compile(self,
                ckpt_dir: str,
                loss_function: str = 'ce',
                optimizer: str = 'adam',
                scheduler: str = 'cosine',
                learnig_rate: float = 1e-4,
                epochs: int = 300,
                local_rank: int = 0):

        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.epochs = epochs
        
        # CUDA 사용 가능 여부 확인 및 디바이스 설정
        if torch.cuda.is_available():
            try:
                self.device = f"cuda:{local_rank}"
                # 실제로 CUDA 디바이스가 작동하는지 테스트
                test_tensor = torch.tensor([1.0]).to(self.device)
                print(f"CUDA 사용 가능: {self.device}")
                print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            except Exception as e:
                print(f"CUDA 초기화 실패: {e}")
                print("CPU 모드로 전환합니다.")
                self.device = "cpu"
        else:
            print("CUDA를 사용할 수 없습니다.")
            print("CPU 모드로 학습을 진행합니다.")
            self.device = "cpu"

        # 디바이스 정보 출력
        print(f"🎯 사용 디바이스: {self.device}")

        # 모델을 디바이스로 이동
        try:
            self.model.to(self.device)
            print("모델이 디바이스로 성공적으로 이동되었습니다.")
        except Exception as e:
            print(f"모델 디바이스 이동 실패: {e}")
            # CPU로 강제 설정
            self.device = "cpu"
            self.model.to(self.device)
            print("강제로 CPU 모드로 전환했습니다.")

        # Define Optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(),
                                        lr=learnig_rate, weight_decay=1e-5)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(params=self.model.parameters(),
                                        lr=learnig_rate, weight_decay=1e-5)
        elif optimizer == 'adamW':
            self.optimizer = optim.AdamW(params=self.model.parameters(),
                                        lr=learnig_rate, weight_decay=1e-5)
        else:
            raise ValueError('Provide a proper optimizer name')

        if scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.0001)

        # Define Loss function
        if loss_function == "ce":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == "mse":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError('Provide a proper loss function name')

        self.compiled = True
        
        # 학습 환경 정보 출력
        print(f"\n 학습 환경 설정 완료:")
        print(f"   디바이스: {self.device}")
        print(f"   옵티마이저: {optimizer}")
        print(f"   학습률: {learnig_rate}")
        print(f"   에포크: {epochs}")
        print(f"   손실함수: {loss_function}")

    def train(self, epoch, data_loader):

        if not self.compiled:
            raise RuntimeError('Training not prepared')

        self._set_learning_phase(True)
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

        for inputs, targets in train_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.model(inputs)
            loss = self.loss_function(pred, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            
            # CPU 사용 시에는 더 자세한 진행 상황 표시
            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            train_bar.set_description('Train Epoch: [{}/{}] [{}], lr: {:.6f}, Loss: {:.4f}'.format(
                epoch, self.epochs, device_info, self.get_lr(self.optimizer), total_loss / total_num))
        return total_loss / total_num

    @torch.no_grad()
    def valid(self, epoch, data_loader):
        pred_list, label_list = [], []
        self._set_learning_phase(False)  # 수정: True -> False (validation 모드)
        total_loss, total_num, valid_bar = 0.0, 0, tqdm(data_loader)

        for inputs, targets in valid_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.model(inputs)

            _, predicted = torch.max(pred.data, 1)
            pred_list += predicted.tolist()
            label_list += targets.tolist()

            loss = self.loss_function(pred, targets)
            valid_acc = accuracy_score(pred_list, label_list)

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            
            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            valid_bar.set_description('Valid Epoch: [{}/{}] [{}], Loss: {:.4f}, Acc: {:.3f}'.format(
                epoch, self.epochs, device_info, total_loss / total_num, valid_acc))
        return total_loss / total_num, valid_acc

    @torch.no_grad()
    def test(self, epoch, data_loader):
        pred_list, label_list = [], []
        self._set_learning_phase(False)
        test_bar = tqdm(data_loader)

        for inputs, targets in test_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.model(inputs)

            _, predicted = torch.max(pred.data, 1)
            pred_list += predicted.tolist()
            label_list += targets.tolist()

            test_acc = accuracy_score(pred_list, label_list)
            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            test_bar.set_description('Test Epoch: [{}/{}] [{}], ACC: {:.3f}'.format(
                epoch, self.epochs, device_info, test_acc))
        return test_acc

    def save(self, epoch, results):
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(self.ckpt_dir + '/log.csv', index_label='epoch')
        
        # 모델 저장 시 디바이스 정보도 함께 저장
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'device': self.device},  # 디바이스 정보 추가
                    self.ckpt_dir + '/model_last.pth')

    def _set_learning_phase(self, train: bool = False):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']