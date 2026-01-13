import torch
import os
import argparse
import json
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np

from torch_trainer.models import build_model, get_gradcam_target_layer
from utils.gradcam import GradCAM, save_gradcam_result

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean

def main(args):
    # 1. Load configuration from class_info.json
    config_path = os.path.join(args.model_path, 'class_info.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name = config['model_name']
    n_class = config['n_class']
    img_size = config['img_size']
    class_names = config['class_names']

    # 2. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(model_name=model_name, pre_trained=False, n_class=n_class)
    
    model_weight_path = os.path.join(args.model_path, 'model_last.pth')
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Model weights not found at {model_weight_path}")

    checkpoint = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 3. Get image paths
    image_paths = glob.glob(os.path.join(args.inference_dir, '*.png'))
    if not image_paths:
        print(f"No PNG images found in {args.inference_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    if args.generate_gradcam:
        os.makedirs(os.path.join(args.output_dir, 'gradcam'), exist_ok=True)


    # 4. Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Grad-CAM setup
    gradcam = None
    if args.generate_gradcam:
        target_layer = get_gradcam_target_layer(model, model_name)
        gradcam = GradCAM(model, target_layer)

    # 5. Inference loop
    results = []
    for img_path in tqdm(image_paths, desc="Running Inference"):
        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Could not read image {img_path}. Skipping. Error: {e}")
            continue

        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        result = {
            "image": os.path.basename(img_path),
            "predicted_class": predicted_class,
            "confidence": confidence_score
        }
        results.append(result)
        print(f"Image: {result['image']}, Prediction: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")

        # Generate and save Grad-CAM
        if gradcam:
            cam, class_idx, _ = gradcam.generate_cam(image_tensor.clone(), class_idx=predicted_idx.item())
            
            # Denormalize image for visualization
            original_image_vis = denormalize(image_tensor.cpu())
            
            overlayed_image, heatmap = gradcam.visualize_cam(original_image_vis, cam)
            
            save_path = os.path.join(args.output_dir, 'gradcam', f"gradcam_{os.path.basename(img_path)}")
            save_gradcam_result(
                original_image=original_image_vis, 
                gradcam_overlay=(overlayed_image, heatmap),
                save_path=save_path,
                predicted_class=class_idx,
                confidence=confidence_score,
                class_names=class_names
            )
    
    # Save results to a JSON file
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nInference complete. Results saved to {results_path}")
    if args.generate_gradcam:
        print(f"Grad-CAM visualizations saved in {os.path.join(args.output_dir, 'gradcam')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for skin detection model.")
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved model directory (containing class_info.json and model_last.pth)')
    parser.add_argument('--inference_dir', type=str, default='inference_set',
                        help='Directory containing PNG images for inference.')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results and Grad-CAM visualizations.')
    parser.add_argument('--generate_gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations for each image.')
    
    args = parser.parse_args()
    main(args)
