# scripts/warmup.py
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
import onnxruntime as ort
from pathlib import Path

print("Warmup start…")

# OCR warmup (downloads models)
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print("PaddleOCR ready")

# InsightFace warmup (downloads antelope/buffalo models on first run)
fa = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready")

# Optional: liveness ONNX warmup if you bundle the model
onnx_path = Path("models/silent_face_80x80.onnx")
if onnx_path.exists():
    ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("Liveness ONNX session ready")
else:
    print("Liveness model not found (skipping).")

print("Warmup done.")
