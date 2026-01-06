import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import wandb
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torch.nn.functional as F

from utils.losses import FocalLoss

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer(object):
    def __init__(self, model, config):
        super(Trainer, self).__init__()
        self.model = model
        self.config = config
        self.compiled = False
        self.early_stopping_patience = config.early_stopping_patience
        self.early_stopping_counter = 0
        self.best_score = None
        self.early_stopping_metric = config.early_stopping_metric
        self.scaler = GradScaler(enabled=self.config.use_amp)

        if self.config.wandb:
            wandb.init(project=self.config.wandb_project,
                       name=self.config.wandb_run_name,
                       config=vars(self.config))
            wandb.watch(self.model)

    def compile(
                self,
                ckpt_dir: str,
                loss_function: str = 'ce',
                optimizer: str = 'adamW',
                scheduler: str = 'cosine',
                learnig_rate: float = 1e-3,
                epochs: int = 50,
                local_rank: int = 0):

        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.epochs = epochs
        
        if torch.cuda.is_available():
            try:
                self.device = f"cuda:{local_rank}"
                test_tensor = torch.tensor([1.0]).to(self.device)
                print(f"CUDA 사용 가능: {self.device}")
                print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            except Exception as e:
                print(f"CUDA 초기화 실패: {e}")
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        if self.device == 'cpu':
            print("CPU 모드로 학습을 진행합니다.")

        self.model.to(self.device)
        print("모델이 디바이스로 성공적으로 이동되었습니다.")

        # Optimizer with new weight_decay
        if optimizer == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=learnig_rate, weight_decay=self.config.weight_decay)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(params=self.model.parameters(), lr=learnig_rate, weight_decay=self.config.weight_decay)
        elif optimizer == 'adamW':
            self.optimizer = optim.AdamW(params=self.model.parameters(), lr=learnig_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError('Provide a proper optimizer name')

        # Scheduler with Warm-up
        if scheduler == 'cosine':
            if self.config.use_warmup:
                print(f"Using Warm-up for {self.config.warmup_epochs} epochs.")
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs - self.config.warmup_epochs, eta_min=1e-5)
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=self.config.warmup_epochs)
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.config.warmup_epochs])
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)
        else:
            self.scheduler = None

        # Loss function selection with Focal Loss
        if self.config.use_focal_loss:
            print(f"Using Focal Loss (gamma={self.config.focal_loss_gamma})")
            self.loss_function = FocalLoss(gamma=self.config.focal_loss_gamma)
        else:
            print(f"Using Cross Entropy Loss (label_smoothing={self.config.label_smoothing})")
            self.loss_function = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)

        self.compiled = True
        
        print(f"\n 학습 환경 설정 완료:")
        print(f"   디바이스: {self.device}")
        print(f"   옵티마이저: {optimizer}")
        print(f"   학습률: {learnig_rate}")
        print(f"   Weight Decay: {self.config.weight_decay}")
        print(f"   에포크: {epochs}")
        print(f"   손실함수: {'Focal Loss (gamma=' + str(self.config.focal_loss_gamma) + ')' if self.config.use_focal_loss else 'Cross Entropy (label_smoothing=' + str(self.config.label_smoothing) + ')'}")
        print(f"   Mixup 사용: {'Yes (alpha=' + str(self.config.mixup_alpha) + ')' if self.config.use_mixup else 'No'}")
        print(f"   Scheduler: {'Cosine with Warm-up (eta_min=1e-5)' if self.config.use_warmup and self.scheduler else 'Cosine (eta_min=1e-5)' if self.scheduler else 'None'}")


    def train(self, epoch, data_loader):
        if not self.compiled:
            raise RuntimeError('Training not prepared')
        
        self._set_learning_phase(True)
        
        if self.config.use_mixup:
            return self._train_mixup_epoch(epoch, data_loader)
        else:
            return self._train_standard_epoch(epoch, data_loader)

    def _train_standard_epoch(self, epoch, data_loader):
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with autocast(enabled=self.config.use_amp):
                pred = self.model(inputs)
                loss = self.loss_function(pred, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            
            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            train_bar.set_description(f'Train Epoch (Standard): [{epoch}/{self.epochs}] [{device_info}], lr: {self.get_lr(self.optimizer):.6f}, Loss: {total_loss / total_num:.4f}')
        
        if self.config.wandb:
            wandb.log({"train_loss": total_loss / total_num}, step=epoch)
        return total_loss / total_num

    def _train_mixup_epoch(self, epoch, data_loader):
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, self.config.mixup_alpha, use_cuda=self.device.startswith('cuda'))

            with autocast(enabled=self.config.use_amp):
                pred = self.model(mixed_inputs)
                loss = mixup_criterion(self.loss_function, pred, targets_a, targets_b, lam)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size

            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            train_bar.set_description(f'Train Epoch (Mixup): [{epoch}/{self.epochs}] [{device_info}], lr: {self.get_lr(self.optimizer):.6f}, Loss: {total_loss / total_num:.4f}')

        if self.config.wandb:
            wandb.log({"train_loss": total_loss / total_num}, step=epoch)
        return total_loss / total_num

    def epoch_step(self):
        if self.scheduler:
            self.scheduler.step()

    @torch.no_grad()
    def valid(self, epoch, data_loader):
        pred_list, label_list = [], []
        self._set_learning_phase(False)
        total_loss, total_num, valid_bar = 0.0, 0, tqdm(data_loader)
        early_stop = False

        for inputs, targets in valid_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast(enabled=self.config.use_amp):
                pred = self.model(inputs)
                loss = self.loss_function(pred, targets)
            _, predicted = torch.max(pred.data, 1)
            pred_list.extend(predicted.cpu().numpy())
            label_list.extend(targets.cpu().numpy())
            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            valid_acc = accuracy_score(label_list, pred_list)
            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            valid_bar.set_description(f'Valid Epoch: [{epoch}/{self.epochs}] [{device_info}], Loss: {total_loss / total_num:.4f}, Acc: {valid_acc:.3f}')
        
        if self.config.wandb:
            wandb.log({"valid_loss": total_loss / total_num, "valid_acc": valid_acc}, step=epoch)

        score = total_loss / total_num if self.early_stopping_metric == 'valid_loss' else valid_acc
        if self.best_score is None or (score < self.best_score and self.early_stopping_metric == 'valid_loss') or (score > self.best_score and self.early_stopping_metric == 'valid_acc'):
            self.best_score = score
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                early_stop = True
        return total_loss / total_num, valid_acc, early_stop

    @torch.no_grad()
    def test(self, epoch, data_loader):
        pred_list, label_list = [], []
        self._set_learning_phase(False)
        test_bar = tqdm(data_loader)
        class_names = data_loader.dataset.classes if hasattr(data_loader.dataset, 'classes') else [str(i) for i in range(self.config.n_class)]

        for inputs, targets in test_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast(enabled=self.config.use_amp):
                pred = self.model(inputs)
            _, predicted = torch.max(pred.data, 1)
            pred_list.extend(predicted.cpu().numpy())
            label_list.extend(targets.cpu().numpy())
            test_acc = accuracy_score(label_list, pred_list)
            device_info = "CPU" if self.device == "cpu" else f"GPU-{self.device}"
            test_bar.set_description(f'Test Epoch: [{epoch}/{self.epochs}] [{device_info}], ACC: {test_acc:.3f}')
        
        cm = confusion_matrix(label_list, pred_list)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        if self.config.wandb:
            wandb.log({"test_acc": test_acc}, step=epoch)
            wandb.log({"confusion_matrix": wandb.Table(dataframe=cm_df.reset_index())})

        return test_acc, cm_df

    def save(self, epoch, results, confusion_matrix=None):
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(os.path.join(self.ckpt_dir, 'log.csv'), index_label='epoch')
        if confusion_matrix is not None:
            confusion_matrix.to_csv(os.path.join(self.ckpt_dir, 'confusion_matrix_test.csv'))
        
        model_path = os.path.join(self.ckpt_dir, 'model_last.pth')
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'device': self.device}, model_path)

        if self.config.wandb:
            wandb.save(model_path)
            if confusion_matrix is not None:
                wandb.save(os.path.join(self.ckpt_dir, 'confusion_matrix_test.csv'))

    def finish(self):
        if self.config.wandb:
            wandb.finish()

    def _set_learning_phase(self, train: bool = False):
        self.model.train(train)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']