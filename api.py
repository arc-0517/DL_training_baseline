"""
FastAPI 기반 피부 진단 API 서버

두 가지 모드 지원:
1. Swagger UI 모드: http://localhost:8000/docs 에서 API 테스트
2. 폴더 감시 모드: watch_folder에 이미지를 넣으면 자동 처리

사용법:
    # 기본 실행 (ONNX 모델 사용)
    python api.py

    # PyTorch 모델 경로 지정
    python api.py --model_path save_results/dataset+skin/model+efficientnet_b1/2026-01-18_17-48-04

    # 폴더 감시 모드
    python api.py --watch

    # 포트 변경
    python api.py --port 8080
"""

import os
import io
import json
import shutil
import argparse
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Transformers (for SegFormer processor)
from transformers import SegformerImageProcessor

# PyTorch model builder
from torch_trainer.models import build_model

# Watchdog for folder monitoring
try:
    from watchdog.observers.polling import PollingObserver as Observer  # WSL/Windows 호환
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("[WARNING] watchdog not installed. Folder watching disabled.")


# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
ONNX_MODEL_DIR = BASE_DIR / "onnx_models"
WATCH_INPUT_DIR = BASE_DIR / "watch_input"
WATCH_OUTPUT_DIR = BASE_DIR / "watch_output"
SEGMENTED_DIR = BASE_DIR / "api_segmented"
UPLOAD_DIR = BASE_DIR / "api_uploads"

# Create directories
for dir_path in [WATCH_INPUT_DIR, WATCH_OUTPUT_DIR, SEGMENTED_DIR, UPLOAD_DIR]:
    dir_path.mkdir(exist_ok=True)

# Segmentation labels
SIMPLIFIED_LABELS = {
    'background': 0, 'skin': 1, 'eyebrow': 2, 'eye': 3, 'eye_g': 4,
    'ear': 5, 'nose': 6, 'mouth': 7, 'neck': 8, 'hair': 9
}

LABEL_MAPPING = {
    0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 0,
    10: 6, 11: 7, 12: 7, 13: 7, 14: 8, 15: 0, 16: 0, 17: 9, 18: 0
}

# Korean translations for class names
CLASS_NAME_KR = {
    "normal": "정상",
    "acne": "여드름",
    "rosacea": "주사",
    "atopic": "아토피",
    "psoriasis": "건선",
    "seborrheic": "지루성"
}


# ============================================================
# Model Loading
# ============================================================
class SkinDetectionModel:
    """피부 진단 모델 클래스 (PyTorch 및 ONNX 지원)"""

    def __init__(self, model_path: Optional[str] = None, use_onnx: bool = True):
        self.model_path = model_path
        self.use_onnx = use_onnx and ONNX_AVAILABLE

        # Models
        self.seg_session = None  # ONNX segmentation
        self.clf_session = None  # ONNX classification
        self.clf_model_pytorch = None  # PyTorch classification
        self.seg_processor = None

        # Model info
        self.img_size = 224
        self.n_class = 6
        self.class_names = []
        self.class_names_kr = []
        self.model_name = "efficientnet_b0"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Image transform for classification
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_models(self):
        """모델 로드"""
        # Load segmentation model (always ONNX)
        self._load_segmentation_model()

        # Load classification model
        if self.model_path:
            self._load_pytorch_model()
        elif self.use_onnx:
            self._load_onnx_classification_model()
        else:
            raise ValueError("Either model_path or ONNX models required")

    def _load_segmentation_model(self):
        """세그멘테이션 모델 로드 (ONNX)"""
        seg_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']

        seg_model_path = ONNX_MODEL_DIR / "seg_model_fp16.onnx"
        if not seg_model_path.exists():
            seg_model_path = ONNX_MODEL_DIR / "seg_model_fp32.onnx"

        if seg_model_path.exists():
            self.seg_session = ort.InferenceSession(str(seg_model_path), providers=seg_providers)
            self.seg_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
            print(f"[OK] Segmentation model loaded: {seg_model_path.name}")
        else:
            raise FileNotFoundError(f"Segmentation model not found in {ONNX_MODEL_DIR}")

    def _load_onnx_classification_model(self):
        """ONNX 분류 모델 로드"""
        clf_providers = ['CPUExecutionProvider']

        clf_model_path = ONNX_MODEL_DIR / "clf_model_fp16.onnx"
        if not clf_model_path.exists():
            clf_model_path = ONNX_MODEL_DIR / "clf_model_fp32.onnx"

        if clf_model_path.exists():
            self.clf_session = ort.InferenceSession(str(clf_model_path), providers=clf_providers)
            print(f"[OK] Classification model loaded: {clf_model_path.name} (ONNX)")

            # Default 6-class names
            self.n_class = 6
            self.class_names = ["psoriasis", "atopic", "acne", "normal", "rosacea", "seborrheic"]
            self.class_names_kr = [f"{CLASS_NAME_KR.get(n, n)} ({n.capitalize()})" for n in self.class_names]
        else:
            raise FileNotFoundError(f"Classification model not found in {ONNX_MODEL_DIR}")

    def _load_pytorch_model(self):
        """PyTorch 분류 모델 로드"""
        model_dir = Path(self.model_path)

        # Load config
        config_path = model_dir / "class_info.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Extract model info
        self.model_name = config.get('model_name', 'efficientnet_b0')
        self.n_class = config.get('n_class', 6)
        self.img_size = config.get('img_size', 224)

        # Get class names from selected_labels or class_names
        if 'selected_labels' in config and config['selected_labels']:
            self.class_names = config['selected_labels']
        elif 'class_names' in config:
            # Parse folder names like "TS_건선_정면" -> "psoriasis"
            raw_names = config['class_names']
            name_map = {"건선": "psoriasis", "아토피": "atopic", "여드름": "acne",
                       "정상": "normal", "주사": "rosacea", "지루": "seborrheic"}
            self.class_names = []
            for raw in raw_names:
                matched = False
                for kr, en in name_map.items():
                    if kr in raw:
                        self.class_names.append(en)
                        matched = True
                        break
                if not matched:
                    self.class_names.append(raw)

        self.class_names_kr = [f"{CLASS_NAME_KR.get(n, n)} ({n.capitalize()})" for n in self.class_names]

        # Update transform with correct image size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Build and load model
        self.clf_model_pytorch = build_model(
            model_name=self.model_name,
            pre_trained=False,
            n_class=self.n_class
        )

        model_weight_path = model_dir / "model_last.pth"
        if not model_weight_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_weight_path}")

        checkpoint = torch.load(model_weight_path, map_location=self.device)
        self.clf_model_pytorch.load_state_dict(checkpoint['state_dict'])
        self.clf_model_pytorch.to(self.device)
        self.clf_model_pytorch.eval()

        print(f"[OK] Classification model loaded: {self.model_name} (PyTorch)")
        print(f"[OK] Classes ({self.n_class}): {self.class_names}")

    def segment_skin(self, image: Image.Image, parts_to_keep: List[str] = ['skin']) -> Image.Image:
        """SegFormer를 사용하여 피부 영역만 추출"""
        pixel_values = self.seg_processor(images=image, return_tensors="np")['pixel_values']
        onnx_inputs = {self.seg_session.get_inputs()[0].name: pixel_values}
        logits = self.seg_session.run(None, onnx_inputs)[0]

        logits_tensor = torch.from_numpy(logits)
        upsampled_logits = torch.nn.functional.interpolate(
            logits_tensor, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        original_parsing = upsampled_logits.argmax(dim=1)[0].numpy()

        simplified_parsing = np.zeros_like(original_parsing)
        for original_label, simplified_label in LABEL_MAPPING.items():
            simplified_parsing[original_parsing == original_label] = simplified_label

        img_array = np.array(image).copy()
        final_mask = np.zeros_like(simplified_parsing, dtype=bool)

        for part_name in parts_to_keep:
            label_id = SIMPLIFIED_LABELS.get(part_name)
            if label_id is not None:
                final_mask |= (simplified_parsing == label_id)

        img_array[~final_mask] = 0
        return Image.fromarray(img_array)

    def predict(self, image: Image.Image) -> dict:
        """피부 상태 예측"""
        image_tensor = self.transform(image).unsqueeze(0)

        if self.clf_model_pytorch is not None:
            # PyTorch inference
            image_tensor = image_tensor.to(self.device)
            with torch.no_grad():
                output = self.clf_model_pytorch(image_tensor)
            output_tensor = output.cpu()
        else:
            # ONNX inference
            image_numpy = image_tensor.numpy()
            onnx_inputs = {self.clf_session.get_inputs()[0].name: image_numpy}
            output = self.clf_session.run(None, onnx_inputs)[0]
            output_tensor = torch.from_numpy(output.astype(np.float32))

        # Handle NaN/Inf values
        if torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any():
            output_tensor = torch.nan_to_num(output_tensor, nan=0.0, posinf=10.0, neginf=-10.0)

        probs = torch.nn.functional.softmax(output_tensor, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

        all_probs = probs[0].tolist()
        all_probs = [0.0 if (p != p) else p for p in all_probs]

        conf_value = confidence.item()
        if conf_value != conf_value:
            conf_value = 0.0

        return {
            "predicted_class": self.class_names_kr[predicted_idx.item()],
            "predicted_class_en": self.class_names[predicted_idx.item()],
            "confidence": round(conf_value, 4),
            "all_probabilities": {
                self.class_names_kr[i]: round(prob, 4) for i, prob in enumerate(all_probs)
            }
        }

    def full_pipeline(self, image: Image.Image, save_segmented: bool = True,
                      output_path: Optional[Path] = None) -> dict:
        """전체 파이프라인: 세그멘테이션 + 예측"""
        segmented_image = self.segment_skin(image, parts_to_keep=['skin'])

        segmented_path = None
        if save_segmented:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = SEGMENTED_DIR / f"segmented_{timestamp}.png"
            segmented_image.save(output_path)
            segmented_path = str(output_path)

        prediction = self.predict(segmented_image)

        return {
            "segmented_image_path": segmented_path,
            **prediction
        }

    def get_model_info(self) -> dict:
        """현재 로드된 모델 정보 반환"""
        return {
            "model_type": "PyTorch" if self.clf_model_pytorch else "ONNX",
            "model_name": self.model_name,
            "n_class": self.n_class,
            "class_names": self.class_names,
            "class_names_kr": self.class_names_kr,
            "img_size": self.img_size,
            "device": self.device,
            "model_path": str(self.model_path) if self.model_path else "ONNX default"
        }


# Global model instance (will be initialized later)
model: Optional[SkinDetectionModel] = None


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="피부 진단 API (Skin Detection API)",
    description="""
## 피부 상태 진단 API

### 기능
1. **세그멘테이션** (`/segment`): SegFormer를 사용하여 피부 영역만 추출
2. **진단** (`/predict`): 피부 상태 예측
3. **전체 분석** (`/analyze`): 세그멘테이션 + 예측 한번에

### 모델 정보
- `/model/info`: 현재 로드된 모델 정보 확인
    """,
    version="2.0.0"
)


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global model
    print("\n" + "="*50)
    print("  피부 진단 API 서버 시작")
    print("="*50)

    if model is not None:
        # 모델이 이미 로드되어 있는지 확인 (--watch 모드에서 미리 로드한 경우)
        if model.seg_session is None:
            model.load_models()

        info = model.get_model_info()
        print(f"\n[MODEL INFO]")
        print(f"  Type: {info['model_type']}")
        print(f"  Name: {info['model_name']}")
        print(f"  Classes: {info['n_class']} - {info['class_names']}")

    print(f"\n[INFO] Swagger UI: http://localhost:8000/docs")
    print("="*50 + "\n")


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "status": "running",
        "message": "피부 진단 API 서버가 실행 중입니다.",
        "docs": "/docs",
        "model_info": "/model/info"
    }


@app.get("/model/info", summary="모델 정보", tags=["Model"])
async def get_model_info():
    """현재 로드된 모델 정보를 반환합니다."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return model.get_model_info()


@app.post("/segment", summary="피부 세그멘테이션", tags=["Segmentation"])
async def segment_image(
    file: UploadFile = File(..., description="얼굴 이미지 파일 (PNG, JPG)"),
    parts: str = Query("skin", description="추출할 부위 (콤마로 구분)")
):
    """SegFormer를 사용하여 이미지에서 피부 영역만 추출합니다."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="PNG 또는 JPG 이미지만 지원됩니다.")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        upload_path = UPLOAD_DIR / f"upload_{timestamp}_{file.filename}"

        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        image = Image.open(upload_path).convert("RGB")
        parts_list = [p.strip() for p in parts.split(",")]
        segmented_image = model.segment_skin(image, parts_to_keep=parts_list)

        output_filename = f"segmented_{timestamp}_{file.filename}"
        output_path = SEGMENTED_DIR / output_filename
        segmented_image.save(output_path)

        return {
            "success": True,
            "message": "세그멘테이션 완료",
            "original_image": str(upload_path),
            "segmented_image": str(output_path),
            "segmented_filename": output_filename,
            "parts_extracted": parts_list
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")


@app.post("/predict", summary="피부 상태 예측", tags=["Prediction"])
async def predict_skin(
    file: UploadFile = File(..., description="피부 이미지 파일")
):
    """피부 이미지를 분석하여 피부 상태를 예측합니다."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="PNG 또는 JPG 이미지만 지원됩니다.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = model.predict(image)

        return {
            "success": True,
            "filename": file.filename,
            **result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")


@app.post("/analyze", summary="전체 분석", tags=["Full Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="얼굴 이미지 파일")
):
    """세그멘테이션 + 예측을 한번에 수행합니다."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="PNG 또는 JPG 이미지만 지원됩니다.")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        upload_path = UPLOAD_DIR / f"upload_{timestamp}_{file.filename}"

        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        image = Image.open(upload_path).convert("RGB")

        output_filename = f"segmented_{timestamp}_{file.filename}"
        output_path = SEGMENTED_DIR / output_filename

        result = model.full_pipeline(image, save_segmented=True, output_path=output_path)

        return {
            "success": True,
            "filename": file.filename,
            "original_image": str(upload_path),
            "segmented_image": result["segmented_image_path"],
            "prediction": {
                "class": result["predicted_class"],
                "class_en": result["predicted_class_en"],
                "confidence": result["confidence"],
                "all_probabilities": result["all_probabilities"]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.get("/segmented/{filename}", summary="세그멘테이션 이미지 조회", tags=["Files"])
async def get_segmented_image(filename: str):
    """저장된 세그멘테이션 이미지를 조회합니다."""
    file_path = SEGMENTED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


# ============================================================
# Folder Watcher (watchdog)
# ============================================================
class ImageWatchHandler(FileSystemEventHandler):
    """폴더 감시 핸들러"""

    def __init__(self, model_instance: SkinDetectionModel):
        self.model = model_instance
        self.processed_files = set()
        self.processing_lock = threading.Lock()

    def _process_image(self, file_path: Path):
        """이미지 처리 로직"""
        with self.processing_lock:
            # 이미 처리된 파일인지 확인
            if str(file_path) in self.processed_files:
                return

            # 파일이 존재하는지 확인
            if not file_path.exists():
                return

            # 파일 확장자 확인
            if file_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                return

            # 파일 복사가 완료될 때까지 대기 (파일 크기 안정화 확인)
            prev_size = -1
            for _ in range(10):  # 최대 5초 대기
                try:
                    curr_size = file_path.stat().st_size
                    if curr_size == prev_size and curr_size > 0:
                        break
                    prev_size = curr_size
                    time.sleep(0.5)
                except:
                    time.sleep(0.5)

            self.processed_files.add(str(file_path))

        print(f"\n[NEW IMAGE] {file_path.name}")

        try:
            image = Image.open(file_path).convert("RGB")
            output_path = WATCH_OUTPUT_DIR / f"result_{file_path.stem}.png"
            result = self.model.full_pipeline(image, save_segmented=True, output_path=output_path)

            result_json_path = WATCH_OUTPUT_DIR / f"result_{file_path.stem}.json"
            with open(result_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "input_image": str(file_path),
                    "segmented_image": result["segmented_image_path"],
                    "prediction": {
                        "class": result["predicted_class"],
                        "class_en": result["predicted_class_en"],
                        "confidence": result["confidence"],
                        "all_probabilities": result["all_probabilities"]
                    },
                    "processed_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)

            print(f"[RESULT] {result['predicted_class']} (confidence: {result['confidence']:.2%})")
            print(f"[SAVED] {result_json_path}")

        except Exception as e:
            print(f"[ERROR] {file_path.name}: {str(e)}")

    def on_created(self, event):
        """파일 생성 이벤트"""
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        self._process_image(file_path)

    def on_modified(self, event):
        """파일 수정 이벤트 (Windows 복사 시 발생)"""
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        self._process_image(file_path)

    def on_moved(self, event):
        """파일 이동 이벤트 (드래그앤드롭 시 발생)"""
        if event.is_directory:
            return
        file_path = Path(event.dest_path)
        self._process_image(file_path)


def start_folder_watcher(model_instance: SkinDetectionModel):
    """폴더 감시 시작"""
    if not WATCHDOG_AVAILABLE:
        print("[ERROR] watchdog not installed.")
        return None

    event_handler = ImageWatchHandler(model_instance)
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_INPUT_DIR), recursive=False)
    observer.start()

    print(f"\n[WATCHING] {WATCH_INPUT_DIR}")
    print(f"[OUTPUT] {WATCH_OUTPUT_DIR}")

    return observer


# ============================================================
# Main
# ============================================================
def main():
    global model

    parser = argparse.ArgumentParser(description="피부 진단 API 서버")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트')
    parser.add_argument('--model_path', type=str, default=None,
                        help='PyTorch 모델 경로 (예: save_results/dataset+skin/model+efficientnet_b1/2026-01-18_17-48-04)')
    parser.add_argument('--use_onnx', action='store_true', default=True,
                        help='ONNX 모델 사용 (기본값)')
    parser.add_argument('--watch', action='store_true', help='폴더 감시 모드 활성화')
    parser.add_argument('--watch-only', action='store_true', help='API 서버 없이 폴더 감시만')

    args = parser.parse_args()

    # Initialize model
    model = SkinDetectionModel(
        model_path=args.model_path,
        use_onnx=args.use_onnx and args.model_path is None
    )

    # Watch-only mode
    if args.watch_only:
        print("\n" + "="*50)
        print("  폴더 감시 모드")
        print("="*50)

        model.load_models()
        observer = start_folder_watcher(model)

        if observer:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        return

    # Start folder watcher in background
    if args.watch:
        # 모델을 미리 로드 (watcher에서 사용하기 위해)
        print("\n[INFO] 폴더 감시 모드: 모델 미리 로드 중...")
        model.load_models()

        def run_watcher():
            time.sleep(2)  # 서버 시작 대기
            start_folder_watcher(model)

        watcher_thread = threading.Thread(target=run_watcher, daemon=True)
        watcher_thread.start()

    # Run FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
