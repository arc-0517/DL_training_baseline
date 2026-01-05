import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn import preprocessing
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def get_transforms(augmentation_type, img_size):
    
    # Always apply base transforms
    base_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if augmentation_type == 'base':
        return transforms.Compose(base_transforms)

    # Define augmentation sets
    geometric_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
    ]
    color_transforms = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]

    # Start with an empty list for augmentation specific transforms
    aug_specific_transforms = []

    if augmentation_type == 'geometric':
        aug_specific_transforms.extend(geometric_transforms)
    elif augmentation_type == 'color':
        aug_specific_transforms.extend(color_transforms)
    elif augmentation_type == 'mixed':
        aug_specific_transforms.extend(geometric_transforms)
        aug_specific_transforms.extend(color_transforms)
    elif augmentation_type == 'randaugment':
        aug_specific_transforms.append(transforms.RandAugment())
    elif augmentation_type == 'autoaugment':
        aug_specific_transforms.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))

    # Combine augmentation specific transforms with base transforms
    return transforms.Compose(aug_specific_transforms + base_transforms)


def get_skin_datasets(data_dir, img_size, augmentation_type='mixed'):
    """
    Returns training and validation datasets for the skin dataset.
    """
    train_transform = get_transforms(augmentation_type, img_size)
    val_transform = get_transforms('base', img_size)

    train_dataset_path = os.path.join(data_dir, 'skin_dataset', 'Training')
    val_dataset_path = os.path.join(data_dir, 'skin_dataset', 'Validation', '01.원천데이터')

    train_dataset = ImageFolder(root=train_dataset_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_dataset_path, transform=val_transform)

    return train_dataset, val_dataset

class DL_dataset(Dataset):
    def __init__(self, dataroot: str, dataname: str, split: str, img_size: int = 224, augmentation_type: str = 'mixed'):

        self.data_dir = dataroot
        self.data_name = dataname
        self.split = split
        self.img_size = img_size
        
        # This class is now only for the 'dogcat' dataset.
        self.input_files = glob.glob(os.path.join(self.data_dir,
                                                  f"{self.data_name}_dataset",
                                                  f"{self.split}set",
                                                  "*jpg"))
        self.target_files = [i.split("\\")[-1].split(".")[0] if "\\" in i 
                           else i.split("/")[-1].split(".")[0] for i in self.input_files]

        le = preprocessing.LabelEncoder()
        self.targets = le.fit_transform(self.target_files)
        self.classes = le.classes_  # 클래스 이름 저장
        
        if split == 'train':
            self.img_trans = get_transforms(augmentation_type, img_size)
        else:
            self.img_trans = get_transforms('base', img_size)

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
