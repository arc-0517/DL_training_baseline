from dataloaders.datasets.my_dataset import DL_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch

def make_data_loaders(config):

    trainset = DL_dataset(dataroot=config.data_dir, dataname=config.data_name, split='train')
    testset = DL_dataset(dataroot=config.data_dir, dataname=config.data_name, split='test')

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

    valid_loader = DataLoader(trainset,
                              batch_size=config.batch_size,
                              sampler=valid_sampler,
                              pin_memory=True,
                              drop_last=True)

    test_loader = DataLoader(testset,
                             batch_size=config.test_batch_size,
                             pin_memory=True)

    return train_loader, valid_loader, test_loader

