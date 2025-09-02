from flask import jsonify
from flask import Flask, render_template, request
from pathlib import Path
import io, base64, uuid, datetime, sqlite3
import numpy as np, cv2
from PIL import Image, ImageOps, ImageFilter
import pytesseract, fitz
from paddleocr import PaddleOCR
import onnxruntime as ort
from insightface.app import FaceAnalysis

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"; UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "kyc_logs.db"
MODELS_DIR = APP_DIR / "models"
SILENT_FACE_ONNX = MODELS_DIR / "silent_face_80x80.onnx"
# PaddleOCR (English), with angle classification on
OCR = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ------------------ DB ------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY,
            ts TEXT,
            user_name TEXT,
            id_file TEXT,
            selfie_file TEXT,
            ocr_name TEXT,
            ocr_address TEXT,
            ocr_dob TEXT,
            liveness_score REAL,
            face_match INTEGER,
            decision TEXT,
            client_ip TEXT
        )
    """)
    # New table for successful verifications only
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verified_people(
            id INTEGER PRIMARY KEY,
            ts TEXT,                -- verification time
            name TEXT,
            address TEXT
        )
    """)
    con.commit()
    con.close()
init_db()

# ------------------ Utilities ------------------
def pil_from_any(path: Path):
    suf = path.suffix.lower()
    if suf in [".jpg",".jpeg",".png"]:
        return Image.open(path).convert("RGB")
    elif suf==".pdf":
        doc = fitz.open(str(path))
        if len(doc)==0: raise ValueError("PDF has no pages")
        page = doc.load_page(0)
        mat = fitz.Matrix(2,2)  # 2x for better OCR
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    else:
        raise ValueError("Unsupported file type")

def pil_from_b64(data_url):
    b64 = data_url.split(",",1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def cv2_from_pil(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def save_image(pil_img: Image.Image, prefix: str) -> Path:
    p = UPLOAD_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    pil_img.save(p, quality=95)
    return p

def ocr_extract_fields(pil_img: Image.Image):
    """
    PaddleOCR-based extraction tuned for Nigerian NIN slips.
    Strategy:
      - OCR all lines (angle cls on).
      - Sort lines by vertical position, then horizontal.
      - Map fields using label heuristics and robust regex.
    Returns: (name, address, dob, raw_text)
    """
    import re

    # Run OCR on RGB ndarray (Paddle accepts np.ndarray directly)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = OCR.ocr(bgr, cls=True)

    lines = []
    if result and result[0]:
        for det in result[0]:
            (pts, (text, score)) = det
            if not text: 
                continue
            xys = np.array(pts, dtype=np.float32)
            cy = float(np.mean(xys[:, 1])); cx = float(np.mean(xys[:, 0]))
            lines.append({"text": text.strip(), "score": float(score), "cx": cx, "cy": cy})

    # Sort top->bottom, then left->right
    lines.sort(key=lambda r: (round(r["cy"]/10), r["cx"]))

    raw_text = "\n".join([l["text"] for l in lines])

    # Normalize helper (uppercase & collapse spaces; keep digits and separators)
    def norm(s: str) -> str:
        s = s.upper()
        s = re.sub(r"[^\w /:\-\.]", " ", s)      # keep letters/numbers/space and a few separators
        s = re.sub(r"\s+", " ", s).strip()
        return s

    nlines = [norm(l["text"]) for l in lines]

    # Heuristics to find label -> value pairs
    def find_value_after(label_keywords):
        """Find the first line containing any label keyword; return the next non-empty line as value."""
        for i, t in enumerate(nlines):
            if any(k in t for k in label_keywords):
                # prefer a value on the *same* line after colon if present
                m = re.search(r":\s*(.+)$", t)
                if m:
                    val = m.group(1).strip()
                    if val and not any(k in val for k in label_keywords):
                        return val
                # otherwise, look ahead
                for j in range(i+1, min(i+4, len(nlines))):
                    v = nlines[j].strip()
                    if v and not any(k in v for k in label_keywords):
                        return v
        return None

    # --- DOB ---
    # Explicit label search first
    dob = find_value_after(["DATE OF BIRTH", "DOB", "BIRTH"])
    # Or pattern anywhere (e.g., "28 JUL 2002" or 30-01-2025)
    if not dob:
        m = re.search(r"\b(\d{1,2}\s+[A-Z]{3}\s+\d{4})\b", raw_text.upper())
        if m: dob = m.group(1)
    if not dob:
        m = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", raw_text.upper())
        if m: dob = m.group(1)

    # --- Names (Surname + Given Names, common on NIN slip) ---
    surname = find_value_after(["SURNAME", "SURNAME/NOM"])
    given   = find_value_after(["GIVEN NAMES", "GIVEN NAMES/PRENOMS", "GIVEN", "FORENAMES"])
    name = None
    # If both available, combine as "Given Surname" (typical display)
    if given and surname:
        # cap length and clean
        name = f"{given.title()} {surname.title()}".strip()
    else:
        # Try a single "Full Name" line
        name = find_value_after(["FULL NAME", "NAME"]) or None
        if name:
            name = name.title()

    # --- Address (not usually on NIN slip, but handle if present) ---
    address = find_value_after(["ADDRESS", "RESIDENCE", "HOME ADDRESS"])
    if address:
        # stitch likely continuation lines if the next lines look address-y
        extras = []
        idxs = [i for i, t in enumerate(nlines) if "ADDRESS" in t]
        start = idxs[0]+1 if idxs else -1
        for j in range(start, min(start+3, len(nlines))):
            v = nlines[j]
            if v and not any(k in v for k in ["ADDRESS","RESIDENCE","HOME ADDRESS","GIVEN","SURNAME","DATE OF BIRTH","DOB","SEX","NIN","NATIONAL IDENTIFICATION NUMBER"]):
                extras.append(v.title())
        if extras:
            address = (address.title() + " " + " ".join(extras)).strip()
        else:
            address = address.title()

    # final cleanups
    if name:
        name = re.sub(r"\s+", " ", name).strip()
    if address:
        address = re.sub(r"\s+", " ", address).strip()
    if dob:
        dob = dob.replace(".", " ").replace("-", " ").replace("/", " ").strip()

    return name, address, dob, raw_text

# ------------------ Models ------------------
class SilentFace:
    """Adapts to ONNX (N,C,H,W) where C in {1,3}."""
    def __init__(self, onnx_path: Path):
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing liveness ONNX at {onnx_path}")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shp = list(inp.shape)
        if len(shp) != 4:
            raise ValueError(f"Unexpected SilentFace input rank (expected 4): {shp}")
        _, C, H, W = [int(s) if isinstance(s, (int, np.integer)) else 0 for s in shp]
        if H == 0: H = 112
        if W == 0: W = 112
        if C not in (1, 3): C = 3
        self.C, self.H, self.W = C, H, W

    def predict_live_score(self, bgr_img: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            h, w = gray.shape
            sz = min(h, w) * 3 // 4
            x = (w - sz)//2; y = (h - sz)//2
            crop_bgr = bgr_img[y:y+sz, x:x+sz]
        else:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            pad = int(0.2*max(w,h))
            x0 = max(0, x-pad); y0 = max(0, y-pad)
            x1 = min(bgr_img.shape[1], x+w+pad); y1 = min(bgr_img.shape[0], y+h+pad)
            crop_bgr = bgr_img[y0:y1, x0:x1]

        if self.C == 3:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            arr = rgb.astype("float32") / 255.0
            arr = np.transpose(arr, (2, 0, 1))
        else:
            g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            arr = g.astype("float32") / 255.0
            arr = arr[np.newaxis, :, :]

        arr = arr[np.newaxis, ...]
        out = self.session.run(None, {self.input_name: arr})[0]
        y = out.squeeze()
        if np.ndim(y) == 0:
            return float(y)
        if y.size == 2:
            live_prob = float(y[0]) / (abs(float(y[0])) + abs(float(y[1])) + 1e-6)
            return (live_prob + 1.0) / 2.0
        return float(np.ravel(y)[0])

class FaceMatcher:
    def __init__(self):
        local_root = MODELS_DIR / "insightface"
        local_root.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], root=str(local_root))
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.last_similarity = None
    @staticmethod
    def _largest_face(faces):
        if not faces: return None
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    def _embedding(self, bgr: np.ndarray):
        faces = self.app.get(bgr)
        f = self._largest_face(faces)
        if f is None or f.embedding is None:
            return None
        return f.embedding.astype("float32")
    def verify(self, id_img_pil: Image.Image, selfie_pil: Image.Image, threshold: float = 0.35) -> bool:
        id_bgr = cv2.cvtColor(np.array(id_img_pil), cv2.COLOR_RGB2BGR)
        live_bgr = cv2.cvtColor(np.array(selfie_pil), cv2.COLOR_RGB2BGR)
        emb1 = self._embedding(id_bgr)
        emb2 = self._embedding(live_bgr)
        if emb1 is None or emb2 is None:
            return False
        sim = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        self.last_similarity = sim
        return sim >= threshold

# Instantiate models (gracefully handle missing)
try:
    LIVENESS = SilentFace(SILENT_FACE_ONNX)
except Exception as e:
    LIVENESS = None
    print("Liveness model load error:", e)

try:
    MATCHER = FaceMatcher()
except Exception as e:
    MATCHER = None
    print("InsightFace init error:", e)

# ------------------ Routes ------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    # Ask only for a *declared* name (optional – OCR name may override)
    declared_name = request.form.get("name","").strip()

    # ID
    id_file = request.files.get("id_file")
    if not id_file:
        return render_template("index.html", result="Missing ID file"), 200
    id_path = UPLOAD_DIR / f"id_{uuid.uuid4().hex[:8]}_{id_file.filename}"
    id_file.save(id_path)

    # Selfie
    selfie_data = request.form.get("selfie_data")
    if not selfie_data:
        return render_template("index.html", result="Missing selfie"), 200
    selfie_img = pil_from_b64(selfie_data)

    # Read ID image
    try:
        id_img = pil_from_any(id_path)
    except Exception as e:
        return render_template("index.html", result=f"Failed to read ID: {e}"), 200

    # OCR (extract fields)
    ocr_name, ocr_addr, ocr_dob, _ = ocr_extract_fields(id_img)
    display_name = ocr_name or declared_name or "(unknown)"

    # Model availability checks
    if LIVENESS is None:
        return render_template("index.html",
                               result="Liveness model missing. Put models/silent_face_80x80.onnx",
                               ocr_name=display_name, ocr_addr=ocr_addr, ocr_dob=ocr_dob), 200
    if MATCHER is None:
        return render_template("index.html",
                               result="Face matcher not ready. Put InsightFace 'buffalo_l' locally.",
                               ocr_name=display_name, ocr_addr=ocr_addr, ocr_dob=ocr_dob), 200

    # Liveness
    live_bgr = cv2_from_pil(selfie_img)
    live_score = float(LIVENESS.predict_live_score(live_bgr))

    # Face match (only if live)
    matched = False
    if live_score >= 0.70:
        matched = MATCHER.verify(id_img, selfie_img, threshold=0.35)

    # Decision & logging
    # NEW
    if live_score >= 0.70 and matched:
        decision = "Verified successfully"
    # store in verified_people
        try:
            con = sqlite3.connect(DB_PATH)
            con.execute(
            "INSERT INTO verified_people(ts, name, address) VALUES (?, ?, ?)",
            (datetime.datetime.now().isoformat(timespec="seconds"),
             ocr_name or declared_name or "(unknown)",
             ocr_addr or "")
            )
            con.commit(); con.close()
        except Exception as e:
            print("verified_people insert error:", e)
    else:
        decision = "couldn't verify face"

    # Always log diagnostic entry (optional)
    try:
        selfie_path = save_image(selfie_img, "selfie")
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO logs(ts,user_name,id_file,selfie_file,ocr_name,ocr_address,ocr_dob,liveness_score,face_match,decision,client_ip) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                datetime.datetime.now().isoformat(timespec="seconds"),
                declared_name,
                str(id_path), str(selfie_path),
                display_name, ocr_addr, ocr_dob,
                live_score, int(bool(matched)), decision, request.remote_addr
            )
        )
        con.commit(); con.close()
    except Exception as e:
        print("Log error:", e)

    # Render with values for modal
    return render_template(
        "index.html",
        result=decision, live_score=f"{live_score:.2f}",
        ocr_name=display_name, ocr_addr=ocr_addr, ocr_dob=ocr_dob
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.post("/admin/verify")
def admin_verify():
    """
    Mark a record as manually verified by an admin.
    Body: { "password": "...", "name": "...", "address": "..." }
    """
    data = request.get_json(silent=True) or {}
    pwd = (data.get("password") or "").strip()
    name = (data.get("name") or "").strip()
    addr = (data.get("address") or "").strip()

    if pwd != "password123":
        return jsonify({"ok": False, "error": "Invalid password"}), 403

    if not name:
        return jsonify({"ok": False, "error": "Missing name"}), 400

    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO verified_people(ts, name, address) VALUES (?, ?, ?)",
            (datetime.datetime.now().isoformat(timespec="seconds"), name, addr)
        )
        con.commit()
        con.close()
        return jsonify({"ok": True})
    except Exception as e:
        print("admin_verify error:", e)
        return jsonify({"ok": False, "error": "DB error"}), 500