from torchvision import models
import torch.nn as nn

def build_model(model_name: str, pre_trained: bool, n_class: int):
    
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pre_trained)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pre_trained)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pre_trained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pre_trained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pre_trained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def get_gradcam_target_layer(model, model_name: str):
    """Grad-CAM을 위한 타겟 레이어 반환"""
    if model_name.startswith("resnet"):
        return model.layer4[-1]  # ResNet의 마지막 conv layer
    elif model_name.startswith("efficientnet"):
        return model.features[-1]  # EfficientNet의 마지막 conv layer
    else:
        raise ValueError(f"Grad-CAM not supported for {model_name}")