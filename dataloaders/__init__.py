from dataloaders.datasets.my_dataset import DL_dataset, get_skin_datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch

def make_data_loaders(config):
    if config.data_name == 'skin':
        train_dataset, valid_dataset = get_skin_datasets(config.data_dir, config.img_size, config.augmentation_type)

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True,
                                  num_workers=config.num_workers)

        valid_loader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False,
                                  num_workers=config.num_workers)

        # Use validation set for testing as well
        test_loader = DataLoader(valid_dataset,
                                 batch_size=config.test_batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=config.num_workers)
                                 
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
                                  num_workers=config.num_workers)

        valid_loader = DataLoader(trainset,
                                  batch_size=config.batch_size,
                                  sampler=valid_sampler,
                                  pin_memory=True,
                                  drop_last=True,
                                  num_workers=config.num_workers)

        test_loader = DataLoader(testset,
                                 batch_size=config.test_batch_size,
                                 pin_memory=True,
                                 num_workers=config.num_workers)

    return train_loader, valid_loader, test_loader
