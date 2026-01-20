from torchvision import models
from torchvision.models import (
    ResNet18_Weights, ResNet50_Weights,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    MobileNet_V2_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
)
import torch
import torch.nn as nn
import timm


def build_model(model_name: str, pre_trained: bool, n_class: int,
                freeze_backbone: bool = False, unfreeze_last_n_blocks: int = 0):

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    elif model_name == "efficientnet_b1":
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.efficientnet_b1(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    elif model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.efficientnet_b2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    # MobileNet V2
    elif model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    # MobileNet V3 Small (가장 경량)
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[3].in_features, n_class)
        )
    # MobileNet V3 Large (V3 중 가장 큼)
    elif model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pre_trained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[3].in_features, n_class)
        )
    elif model_name in ["vit_b_16", "vit_tiny_patch16_224"]:
        model = timm.create_model(model_name, pretrained=pre_trained)
        model.head = nn.Linear(model.head.in_features, n_class)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

def get_gradcam_target_layer(model, model_name: str):
    """Grad-CAM을 위한 타겟 레이어 반환"""
    if model_name.startswith("resnet"):
        return model.layer4[-1]  # ResNet의 마지막 conv layer
    elif model_name.startswith("efficientnet"):
        return model.features[-1]  # EfficientNet의 마지막 conv layer
    elif model_name.startswith("mobilenet"):
        return model.features[-1]  # MobileNet의 마지막 conv layer
    elif model_name.startswith("vit"):
        return model.blocks[-1].norm1  # ViT의 마지막 블록의 norm layer
    else:
        raise ValueError(f"Grad-CAM not supported for {model_name}")
