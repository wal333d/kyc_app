# scripts/warmup.py
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
import onnxruntime as ort
from pathlib import Path

print("Warmup start…")

# OCR warmup (downloads models)
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print("PaddleOCR ready")

from insightface.app import FaceAnalysis

print("Warmup start…")

# PaddleOCR warmup (leave yours as-is)
from paddleocr import PaddleOCR
_ = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print("PaddleOCR ready")

# InsightFace warmup (switch to buffalo_l)
fa = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
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
