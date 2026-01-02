# utils/gradcam.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
                
        self.forward_hook = self.target_layer.register_forward_hook(self.forward_hook_fn)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.backward_hook_fn)
    
    def forward_hook_fn(self, module, input, output):
        self.activations = output
    
    def backward_hook_fn(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __del__(self):        
        if hasattr(self, 'forward_hook'):
            self.forward_hook.remove()
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Grad-CAM 생성
        Args:
            input_tensor: 입력 이미지 텐서 (1, C, H, W)
            class_idx: 타겟 클래스 인덱스 (None이면 예측된 클래스 사용)
        Returns:
            cam: Grad-CAM 히트맵, predicted_class, confidence
        """
        self.model.eval()
        
        # gradient 초기화
        self.gradients = None
        self.activations = None
        
        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 예측 확률 계산
        probs = F.softmax(output, dim=1)
        confidence = probs[0, class_idx].item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Grad-CAM 계산
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check if target layer is correct.")
        
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU 적용
        
        # Resize to input size
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), 
                          mode='bilinear', align_corners=False)
        
        # Normalize
        cam = cam.squeeze()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # gradient 정리
        input_tensor.requires_grad_(False)
        
        return cam.detach().cpu().numpy(), class_idx, confidence

    def visualize_cam(self, original_image, cam, alpha=0.4):
        """
        Grad-CAM 시각화
        Args:
            original_image: 원본 이미지 (PIL Image, numpy array, 또는 torch.Tensor)
            cam: Grad-CAM 히트맵
            alpha: 투명도
        Returns:
            overlayed_image: 오버레이된 이미지
        """
        # 이미지를 numpy array로 변환
        if isinstance(original_image, torch.Tensor):
            # 텐서를 numpy로 변환
            if original_image.dim() == 4:  # (1, C, H, W)
                img_np = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            elif original_image.dim() == 3:  # (C, H, W)
                img_np = original_image.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = original_image.cpu().numpy()
            
            # 0-1 범위로 클리핑하고 0-255로 변환
            img_np = np.clip(img_np, 0, 1)
            original_image = (img_np * 255).astype(np.uint8)
        elif isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        else:
            # 이미 numpy array인 경우
            if original_image.max() <= 1:
                original_image = (original_image * 255).astype(np.uint8)
        
        # CAM을 컬러맵으로 변환
        heatmap = cm.jet(cam)[:, :, :3]  # RGB만 사용
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # 원본 이미지 크기에 맞게 조정
        if original_image.shape[:2] != cam.shape:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # 오버레이
        overlayed = original_image * (1 - alpha) + heatmap * alpha
        overlayed = overlayed.astype(np.uint8)
        
        return overlayed, heatmap

def save_gradcam_result(original_image, gradcam_overlay, save_path, predicted_class, confidence, class_names):
    """Grad-CAM 결과를 이미지로 저장"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지 처리
    if isinstance(original_image, torch.Tensor):
        # 텐서를 numpy로 변환 (배치 차원 제거)
        if original_image.dim() == 4:  # (1, C, H, W)
            img_np = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        elif original_image.dim() == 3:  # (C, H, W)
            img_np = original_image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = original_image.cpu().numpy()
        
        # 이미 0-1 범위로 정규화된 상태이므로 클리핑만
        img_np = np.clip(img_np, 0, 1)
    else:
        # numpy array인 경우
        img_np = original_image
        if img_np.max() > 1:
            img_np = img_np / 255.0
    
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM 히트맵
    axes[1].imshow(gradcam_overlay[1], cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # 오버레이 결과
    axes[2].imshow(gradcam_overlay[0])
    axes[2].set_title(f'Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.3f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()