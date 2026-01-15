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


from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Class labels - order must match folder names in both Training and Validation
LABELS = ['seborrheic', 'rosacea', 'normal', 'acne', 'atopic', 'psoriasis']

class SkinDataset(Dataset):
    """Custom dataset for skin condition classification with new folder structure."""

    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Root directory containing skin_dataset folder
            split: 'train' or 'val'
            transform: torchvision transforms to apply
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.classes = LABELS

        # Determine folder path based on split
        if split == 'train':
            base_path = os.path.join(data_dir, 'skin_dataset', 'Training', '01.원천데이터')
        else:  # val
            base_path = os.path.join(data_dir, 'skin_dataset', 'Validation', '01.원천데이터')

        # Load all image paths and labels directly from folder names
        for label_idx, label_name in enumerate(LABELS):
            folder_path = os.path.join(base_path, label_name)

            # Check if folder exists
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found: {folder_path}")
                continue

            # Find all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files = glob.glob(os.path.join(folder_path, ext))
                for img_path in image_files:
                    self.samples.append((img_path, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {base_path}")

        print(f"Loaded {len(self.samples)} images for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Return class names"""
        return self.classes

def get_skin_datasets(data_dir, img_size, augmentation_type='mixed', valid_ratio=0.2, random_state=42):
    """
    Returns training and validation datasets for the skin dataset.
    Loads data from Training folder and splits it into train/val with fixed random seed.

    Args:
        data_dir: Root directory containing skin_dataset folder
        img_size: Image size for transforms
        augmentation_type: Type of data augmentation for training
        valid_ratio: Ratio of validation data (default: 0.2)
        random_state: Random seed for reproducible split (default: 42)
    """
    train_transform = get_transforms(augmentation_type, img_size)
    val_transform = get_transforms('base', img_size)

    # Load all data from Training folder (without transform first)
    full_dataset = SkinDataset(data_dir=data_dir, split='train', transform=None)

    # Get all sample indices
    indices = list(range(len(full_dataset)))

    # Get labels for stratified split
    labels = [label for _, label in full_dataset.samples]

    # Perform stratified train/val split with fixed random seed
    train_indices, val_indices = train_test_split(
        indices,
        test_size=valid_ratio,
        random_state=random_state,
        stratify=labels  # Ensure balanced class distribution
    )

    # Create train dataset with augmentation
    train_dataset = SkinDataset(data_dir=data_dir, split='train', transform=train_transform)
    train_dataset.samples = [full_dataset.samples[i] for i in train_indices]

    # Create validation dataset without augmentation
    val_dataset = SkinDataset(data_dir=data_dir, split='train', transform=val_transform)
    val_dataset.samples = [full_dataset.samples[i] for i in val_indices]

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
