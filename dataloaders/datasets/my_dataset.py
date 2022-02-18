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
    def __init__(self, dataroot:str, dataname:str, split:str):

        self.data_dir = dataroot
        self.data_name = dataname
        self.split = split
        self.input_files = glob.glob(os.path.join(self.data_dir,
                                                  f"{self.data_name}_dataset",
                                                  f"{self.split}set",
                                                  "*jpg"))
        self.target_files = [i.split("\\")[-1].split(".")[0] for i in self.input_files]

        le = preprocessing.LabelEncoder()
        self.targets = le.fit_transform(self.target_files)
        self.img_trans = transforms.Compose([transforms.Resize((64,64)),
                                             transforms.ToTensor()])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        sample, target = self.input_files[idx], self.targets[idx]
        sample_img = Image.open(sample).convert('RGB')
        imt_rgb = self.img_trans(sample_img)

        return imt_rgb, target



if __name__ == '__main__':

    config = TrainConfig.parse_arguments()
    trainset = DL_dataset(dataroot=config.data_dir, dataname=config.data_name, split='train')

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