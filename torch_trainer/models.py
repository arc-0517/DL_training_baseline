from torchvision import models
import torch.nn as nn
import timm

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
    # MobileNet V2
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pre_trained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, n_class)
        )
    # MobileNet V3 Small (가장 경량)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pre_trained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[3].in_features, n_class)
        )
    # MobileNet V3 Large (V3 중 가장 큼)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pre_trained)
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
