import os
import torch
import glob
from configs import TrainConfig
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing

import numpy as np

import warnings
warnings.filterwarnings("ignore")

class DL_dataset(Dataset):
    def __init__(self, dataroot: str, dataname: str, split: str, img_size: int = 224):

        self.data_dir = dataroot
        self.data_name = dataname
        self.split = split
        self.img_size = img_size
        
        # 데이터셋에 따라 다른 파일 패턴 사용
        if dataname == 'skin':
            # 피부 데이터의 경우 폴더 구조가 다를 수 있음
            self.input_files = []
            dataset_path = os.path.join(self.data_dir, f"{self.data_name}_dataset", f"{self.split}set")
            
            # 클래스별 폴더 구조인 경우
            if os.path.exists(dataset_path):
                for class_folder in os.listdir(dataset_path):
                    class_path = os.path.join(dataset_path, class_folder)
                    if os.path.isdir(class_path):
                        for img_file in os.listdir(class_path):
                            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                self.input_files.append(os.path.join(class_path, img_file))
                
                # 클래스 라벨 생성 (폴더명 기반)
                self.target_files = []
                for file_path in self.input_files:
                    class_name = os.path.basename(os.path.dirname(file_path))
                    self.target_files.append(class_name)
            else:
                # 일반적인 파일명 기반 구조
                self.input_files = glob.glob(os.path.join(self.data_dir,
                                                          f"{self.data_name}_dataset",
                                                          f"{self.split}set",
                                                          "*"))
                # 확장자 필터링
                self.input_files = [f for f in self.input_files 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                self.target_files = [os.path.basename(i).split(".")[0] for i in self.input_files]
        else:
            # 기존 dogcat, santa 데이터셋 처리
            self.input_files = glob.glob(os.path.join(self.data_dir,
                                                      f"{self.data_name}_dataset",
                                                      f"{self.split}set",
                                                      "*jpg"))
            self.target_files = [i.split("\\")[-1].split(".")[0] if "\\" in i 
                               else i.split("/")[-1].split(".")[0] for i in self.input_files]

        le = preprocessing.LabelEncoder()
        self.targets = le.fit_transform(self.target_files)
        self.classes = le.classes_  # 클래스 이름 저장
        
        # 이미지 변환 설정
        if split == 'train':
            self.img_trans = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_trans = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        sample, target = self.input_files[idx], self.targets[idx]
        sample_img = Image.open(sample).convert('RGB')
        imt_rgb = self.img_trans(sample_img)

        return imt_rgb, target

    def get_class_names(self):
        """클래스 이름 반환"""
        return self.classes.tolist()


if __name__ == '__main__':

    config = TrainConfig.parse_arguments()
    trainset = DL_dataset(dataroot=config.data_dir, dataname=config.data_name, 
                         split='train', img_size=config.img_size)

    # Creating data indices for training and validation splits:
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.valid_ratio * dataset_size))
    if config.shuffle_dataset:
        np.random.seed(config.random_state)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(trainset,
                              batch_size=config.batch_size,
                              sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True)

    samples, targets = next(iter(train_loader))