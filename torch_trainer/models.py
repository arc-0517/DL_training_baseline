import torchvision
from torchvision import models
import torch.nn as nn

def build_model(model_name:str, pre_trained:bool, n_class:int):

    if model_name == "resnet18":
        model = models.resnet50(pretrained=pre_trained)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    return model

