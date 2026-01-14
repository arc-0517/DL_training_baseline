from dataloaders.datasets.my_dataset import DL_dataset, get_skin_datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from utils.reproducibility import get_worker_init_fn

import os
from torchvision.datasets import ImageFolder
from dataloaders.datasets.my_dataset import DL_dataset, get_skin_datasets, get_transforms
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from utils.reproducibility import get_worker_init_fn

def make_data_loaders(config):
    # Worker 초기화 함수 생성 (재현성을 위해)
    worker_init_fn = get_worker_init_fn(config.random_state)

    if config.data_name == 'skin':
        train_dataset, valid_dataset = get_skin_datasets(
            data_dir=config.data_dir,
            img_size=config.img_size,
            augmentation_type=config.augmentation_type,
            valid_ratio=config.valid_ratio,
            random_state=config.random_state,
            shuffle=config.shuffle_dataset
        )

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True, # Shuffle the subset of training data
                                  pin_memory=True,
                                  drop_last=True,
                                  num_workers=config.num_workers,
                                  worker_init_fn=worker_init_fn)

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False,
                                  num_workers=config.num_workers,
                                  worker_init_fn=worker_init_fn)

        # Create a separate test dataset from the validation folder
        test_transform = get_transforms('base', config.img_size)
        test_dataset_path = os.path.join(config.data_dir, 'skin_dataset', 'Validation', '01.원천데이터')
        test_dataset = ImageFolder(root=test_dataset_path, transform=test_transform)
        
        test_loader = DataLoader(test_dataset,
                                 batch_size=config.test_batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=config.num_workers,
                                 worker_init_fn=worker_init_fn)
                                 
        # Attach a simple method to the loader to get class names
        def get_class_names_func():
            return train_dataset.classes
        train_loader.dataset.get_class_names = get_class_names_func

    else: # existing dogcat logic
        trainset = DL_dataset(dataroot=config.data_dir, dataname=config.data_name, split='train', augmentation_type=config.augmentation_type)
        testset = DL_dataset(dataroot=config.data_dir, dataname=config.data_name, split='test', augmentation_type='base')

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
                                  drop_last=True,
                                  num_workers=config.num_workers,
                                  worker_init_fn=worker_init_fn)

        valid_loader = DataLoader(trainset,
                                  batch_size=config.batch_size,
                                  sampler=valid_sampler,
                                  pin_memory=True,
                                  drop_last=True,
                                  num_workers=config.num_workers,
                                  worker_init_fn=worker_init_fn)

        test_loader = DataLoader(testset,
                                 batch_size=config.test_batch_size,
                                 pin_memory=True,
                                 num_workers=config.num_workers,
                                 worker_init_fn=worker_init_fn)

    return train_loader, valid_loader, test_loader
