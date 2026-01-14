import torch
import os
import argparse
import json
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np

# Conditional imports
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    SegformerImageProcessor, SegformerForSemanticSegmentation = None, None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from torch_trainer.models import build_model, get_gradcam_target_layer
from utils.gradcam import GradCAM, save_gradcam_result

# --- Segmentation Label Definitions ---
SIMPLIFIED_LABELS = {
    'background': 0, 'skin': 1, 'eyebrow': 2, 'eye': 3, 'eye_g': 4,
    'ear': 5, 'nose': 6, 'mouth': 7, 'neck': 8, 'hair': 9
}

LABEL_MAPPING = {
    0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 0,
    10: 6, 11: 7, 12: 7, 13: 7, 14: 8, 15: 0, 16: 0, 17: 9, 18: 0
}
# --- End of Segmentation Label Definitions ---


def parse_face_pytorch(image, processor, model, device):
    """Parses the face to create a simplified segmentation map using PyTorch."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    original_parsing = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    simplified_parsing = np.zeros_like(original_parsing)
    for original_label, simplified_label in LABEL_MAPPING.items():
        simplified_parsing[original_parsing == original_label] = simplified_label
        
    return simplified_parsing



def mask_image(image, parsing_map, parts_to_keep):
    """Masks the image, keeping only the specified parts."""
    img_array = np.array(image).copy()
    
    final_mask = np.zeros_like(parsing_map, dtype=bool)
    for part_name in parts_to_keep:
        label_id = SIMPLIFIED_LABELS.get(part_name)
        if label_id is not None:
            final_mask |= (parsing_map == label_id)
            
    img_array[~final_mask] = 0
    return Image.fromarray(img_array)


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean

def main(args):
    # --- Argument and Library Checks ---
    if args.use_onnx and ort is None:
        raise ImportError("onnxruntime is not installed. Please install it to use ONNX models.")
    if args.use_segmentation and SegformerImageProcessor is None:
        raise ImportError("transformers is not installed. Please install it to use segmentation.")
    if args.use_onnx and args.generate_gradcam:
        print("[WARNING] Grad-CAM is not supported for ONNX models. Disabling Grad-CAM.")
        args.generate_gradcam = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model Loading ---
    clf_session, seg_session, seg_processor = None, None, None
    clf_model_pytorch, seg_model_pytorch = None, None
    class_names, img_size = [], 224

    if args.use_onnx:
        # ONNX Path
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        clf_model_path = os.path.join(args.onnx_model_dir, "clf_model.onnx")
        if not os.path.exists(clf_model_path):
            raise FileNotFoundError(f"ONNX classification model not found at {clf_model_path}. Please run the conversion script.")
        clf_session = ort.InferenceSession(clf_model_path, providers=providers)
        print(f"Loaded ONNX classification model: {os.path.basename(clf_model_path)}")

        if args.use_segmentation:
            seg_model_path = os.path.join(args.onnx_model_dir, "seg_model.onnx")
            if not os.path.exists(seg_model_path):
                raise FileNotFoundError(f"ONNX segmentation model not found at {seg_model_path}. Please run the conversion script.")
            seg_session = ort.InferenceSession(seg_model_path, providers=providers)
            print(f"Loaded ONNX segmentation model: {os.path.basename(seg_model_path)}")
            seg_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")

        # Load config from original pytorch model dir to get class names, etc.
        config_path = os.path.join(args.model_path, 'class_info.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        class_names, img_size = config['class_names'], config['img_size']

    else:
        # PyTorch Path
        config_path = os.path.join(args.model_path, 'class_info.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        class_names, img_size, model_name = config['class_names'], config['img_size'], config['model_name']
        
        clf_model_pytorch = build_model(model_name=model_name, pre_trained=False, n_class=config['n_class'])
        model_weight_path = os.path.join(args.model_path, 'model_last.pth')
        checkpoint = torch.load(model_weight_path, map_location=device)
        clf_model_pytorch.load_state_dict(checkpoint['state_dict'])
        clf_model_pytorch.to(device)
        clf_model_pytorch.eval()
        print(f"Loaded PyTorch classification model: {model_weight_path}")

        if args.use_segmentation:
            seg_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
            seg_model_pytorch = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device)
            seg_model_pytorch.eval()
            print("Loaded PyTorch segmentation model.")

    # --- Prepare for Inference ---
    image_paths = glob.glob(os.path.join(args.inference_dir, '*.png'))
    if not image_paths:
        print(f"No PNG images found in {args.inference_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    if args.generate_gradcam:
        os.makedirs(os.path.join(args.output_dir, 'gradcam'), exist_ok=True)
    if args.use_segmentation:
        os.makedirs(os.path.join(args.output_dir, 'segmented_images'), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    gradcam = None
    if args.generate_gradcam and clf_model_pytorch:
        target_layer = get_gradcam_target_layer(clf_model_pytorch, model_name)
        gradcam = GradCAM(clf_model_pytorch, target_layer)

    # --- Inference Loop ---
    results = []
    for img_path in tqdm(image_paths, desc="Running Inference"):
        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Could not read image {img_path}. Skipping. Error: {e}")
            continue

        processed_image_pil = image_pil
        if args.use_segmentation:
            parsing_map = None
            if args.use_onnx:
                pixel_values = seg_processor(images=image_pil, return_tensors="np")['pixel_values']
                onnx_inputs = {seg_session.get_inputs()[0].name: pixel_values}
                logits = seg_session.run(None, onnx_inputs)[0]
                logits_tensor = torch.from_numpy(logits)
                upsampled_logits = torch.nn.functional.interpolate(
                    logits_tensor, size=image_pil.size[::-1], mode="bilinear", align_corners=False
                )
                original_parsing = upsampled_logits.argmax(dim=1)[0].numpy()
                
                simplified_parsing = np.zeros_like(original_parsing)
                for original_label, simplified_label in LABEL_MAPPING.items():
                    simplified_parsing[original_parsing == original_label] = simplified_label
                parsing_map = simplified_parsing
            else:
                parsing_map = parse_face_pytorch(image_pil, seg_processor, seg_model_pytorch, device)
            
            processed_image_pil = mask_image(image_pil, parsing_map, args.segmentation_parts)
            
            seg_save_path = os.path.join(args.output_dir, 'segmented_images', f"segmented_{os.path.basename(img_path)}")
            processed_image_pil.save(seg_save_path)

        image_tensor = transform(processed_image_pil).unsqueeze(0)
        
        # Get prediction
        if args.use_onnx:
            input_array = image_tensor.numpy()
            onnx_inputs = {clf_session.get_inputs()[0].name: input_array}
            output = clf_session.run(None, onnx_inputs)[0]
            output = torch.from_numpy(output) # Convert to tensor for softmax
        else:
            with torch.no_grad():
                output = clf_model_pytorch(image_tensor.to(device))

        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

        result = {
            "image": os.path.basename(img_path),
            "predicted_class": class_names[predicted_idx.item()],
            "confidence": confidence.item()
        }
        results.append(result)
        print(f"Image: {result['image']}, Prediction: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")

        if gradcam:
            cam, class_idx, _ = gradcam.generate_cam(image_tensor.to(device).clone(), class_idx=predicted_idx.item())
            original_image_vis = denormalize(image_tensor.cpu())
            overlayed_image, heatmap = gradcam.visualize_cam(original_image_vis, cam)
            save_path = os.path.join(args.output_dir, 'gradcam', f"gradcam_{os.path.basename(img_path)}")
            save_gradcam_result(
                original_image=original_image_vis, gradcam_overlay=(overlayed_image, heatmap),
                save_path=save_path, predicted_class=class_idx, confidence=result['confidence'],
                class_names=class_names
            )
    
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nInference complete. Results saved to {results_path}")
    if args.use_segmentation:
        print(f"Segmented images saved in {os.path.join(args.output_dir, 'segmented_images')}")
    if args.generate_gradcam:
        print(f"Grad-CAM visualizations saved in {os.path.join(args.output_dir, 'gradcam')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference script for skin detection model with optional segmentation and ONNX support.")
    
    # Model Selection
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved PyTorch classification model directory (used for configs).')
    parser.add_argument('--use_onnx', action='store_true',
                        help='Use ONNX models for inference instead of PyTorch.')
    parser.add_argument('--onnx_model_dir', type=str, default='onnx_models',
                        help='Directory containing the ONNX model files.')

    # Input/Output
    parser.add_argument('--inference_dir', type=str, default='inference_set',
                        help='Directory containing PNG images for inference.')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results.')

    # Features
    parser.add_argument('--generate_gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations (PyTorch only).')
    parser.add_argument('--use_segmentation', action='store_true',
                        help='Enable face segmentation to mask parts of the image before classification.')
    parser.add_argument('--segmentation_parts', nargs='+', default=['skin'],
                        help="List of parts to keep after segmentation (e.g., 'skin', 'hair').")
    
    args = parser.parse_args()
    main(args)
