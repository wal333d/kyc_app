# KYC Web App (Flask + PyMuPDF)
- Upload ID (PDF/JPG/PNG)
- Webcam selfie capture
- Liveness (SilentFace ONNX – placeholder)
- Face match (InsightFace ArcFace)
- OCR (Tesseract)
- SQLite logs

## Run
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
sudo apt install -y tesseract-ocr libtesseract-dev
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5000
python3 view_db.py --table verified_people --limit 50
python3 view_db.py --table logs --limit 100000 --csv logs.csv