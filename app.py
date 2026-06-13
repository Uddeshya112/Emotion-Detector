"""
Emotion Detection – Flask Backend
==================================
Endpoints
  GET  /           → serve index.html
  POST /predict    → image upload → JSON prediction
  POST /webcam     → base64 frame → JSON prediction
  GET  /health     → health-check
"""

import os
import io
import base64
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from PIL import Image

from predictor import EmotionPredictor

# ── App configuration ─────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = str(BASE_DIR / "best_emotion_model.pth")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"]      = str(UPLOAD_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model (singleton) ────────────────────────────────────────────────────

try:
    predictor = EmotionPredictor(model_path=MODEL_PATH)
    MODEL_OK = True
except Exception as exc:
    print(f"[ERROR] Failed to initialise predictor: {exc}")
    predictor = None
    MODEL_OK = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def error(message: str, code: int = 400):
    return jsonify({"success": False, "error": message}), code


def prediction_response(result: dict):
    return jsonify({
        "success":    True,
        "emotion":    result["emotion"],
        "confidence": result["confidence"],
        "emoji":      result["emoji"],
        "all_scores": result["all_scores"],
    })

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "model_ready": MODEL_OK,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Accept multipart image upload, return emotion prediction."""
    if not MODEL_OK or predictor is None:
        return error("Model not loaded – check server logs.", 503)

    if "image" not in request.files:
        return error("No image file provided. Field name must be 'image'.")

    file = request.files["image"]

    if file.filename == "":
        return error("No file selected.")

    if not allowed_file(file.filename):
        return error(
            f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}."
        )

    try:
        # Read raw bytes (avoid saving to disk unless needed for debugging)
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return error("Uploaded file is empty.")

        # Quick validity check
        Image.open(io.BytesIO(img_bytes)).verify()

        result = predictor.predict(img_bytes)
        return prediction_response(result)

    except Exception:
        traceback.print_exc()
        return error("Could not process image. Ensure it is a valid image file.", 422)


@app.route("/webcam", methods=["POST"])
def webcam():
    """Accept a base64-encoded image frame, return emotion prediction."""
    if not MODEL_OK or predictor is None:
        return error("Model not loaded – check server logs.", 503)

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return error("JSON body with 'image' (base64) is required.")

    b64_string: str = data["image"]

    # Strip data-URI header if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        if not b64_string or len(b64_string) == 0:
            return error("Empty image data provided.")
            
        img_bytes = base64.b64decode(b64_string)
        
        if len(img_bytes) == 0:
            return error("Decoded image is empty.")
        
        result    = predictor.predict(img_bytes)
        return prediction_response(result)

    except base64.binascii.Error as e:
        print(f"[ERROR] Base64 decode error: {e}")
        return error("Invalid base64 encoding.")
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Webcam prediction error: {e}")
        return error(f"Could not process webcam frame: {str(e)}", 422)


# ── 413 handler (file too large) ──────────────────────────────────────────────

@app.errorhandler(413)
def too_large(_e):
    return error(f"File exceeds {MAX_CONTENT_LENGTH // (1024*1024)} MB limit.", 413)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
