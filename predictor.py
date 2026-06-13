"""
Emotion Detection Predictor
Loads a ResNet18 model and predicts emotions from facial images.
Includes face detection via OpenCV Haar Cascade for accurate cropping.
"""

import io
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ── Emotion classes (must match training order) ───────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_EMOJIS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😊",
    "Neutral":  "😐",
    "Sad":      "😢",
    "Surprise": "😲",
}

# ── Image preprocessing pipeline ─────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),   # convert face to 3-ch
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def build_model(num_classes: int = 7) -> nn.Module:
    """Create a ResNet18 matching the training architecture."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    # Must match the classifier head used in train_pytorch_fast.py
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


# ── Face detector (OpenCV Haar Cascade) ──────────────────────────────────────
# Use the bundled frontal-face cascade that ships with OpenCV
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)


class EmotionPredictor:
    """
    Singleton-style predictor that loads the model once and exposes
    a `predict` method for images (PIL or bytes).
    Automatically detects and crops the face region before prediction.
    """

    def __init__(self, model_path: str = "best_emotion_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self, path: str) -> nn.Module:
        model = build_model(num_classes=len(EMOTIONS))

        if not os.path.isfile(path):
            # ── DEMO MODE ─────────────────────────────────────────────────────
            # No trained weights found → return randomly initialised model so
            # the application still runs and returns plausible-looking output.
            print(
                f"[WARNING] Model file '{path}' not found. "
                "Running in DEMO mode with random weights."
            )
            model = model.to(self.device)
            return model

        # Load weights (handles DataParallel / plain state-dicts)
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # assume raw state-dict

        # Strip DataParallel "module." prefix if present
        stripped = {k.replace("module.", ""): v for k, v in state_dict.items()}

        try:
            model.load_state_dict(stripped, strict=True)
        except RuntimeError:
            # Fall back to non-strict load (e.g. different head size)
            model.load_state_dict(stripped, strict=False)
            print("[WARNING] Non-strict model load – some layers may differ.")

        model = model.to(self.device)
        print(f"[INFO] Model loaded from '{path}' on {self.device}.")
        return model

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, image_source) -> dict:
        """
        Parameters
        ----------
        image_source : PIL.Image | bytes | BytesIO

        Returns
        -------
        {
            "emotion":    "Happy",
            "confidence": 92.45,
            "emoji":      "😊",
            "all_scores": {"Angry": 1.2, "Happy": 92.45, ...}
        }
        """
        pil_image = self._to_pil(image_source)
        # Crop to face region so the model sees the same kind of
        # tightly-cropped face it was trained on (FER-2013 style).
        pil_image = self._crop_face(pil_image)
        tensor = self._preprocess(pil_image)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze()

        scores = {EMOTIONS[i]: round(float(probabilities[i]) * 100, 2)
                  for i in range(len(EMOTIONS))}

        top_idx = int(torch.argmax(probabilities).item())
        emotion = EMOTIONS[top_idx]
        confidence = scores[emotion]

        return {
            "emotion":    emotion,
            "confidence": confidence,
            "emoji":      EMOTION_EMOJIS[emotion],
            "all_scores": scores,
        }

    # ── face detection ────────────────────────────────────────────────────────

    @staticmethod
    def _crop_face(img: Image.Image) -> Image.Image:
        """
        Detect the largest face in the image and return a cropped + padded
        PIL image of just that face.  Falls back to the full image if no
        face is found.
        """
        # Convert PIL → BGR numpy array for OpenCV
        img_rgb = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # No face found — use the full image as-is
            print("[INFO] No face detected; using full image.")
            return img

        # Pick the largest face by area
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        # Add a small padding (10 %) so we don't clip eyebrows / chin
        pad_x = int(w * 0.10)
        pad_y = int(h * 0.10)
        H, W  = img_rgb.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)

        face_crop = img_rgb[y1:y2, x1:x2]
        print(f"[INFO] Face detected at ({x},{y}) size {w}x{h}; cropped to ({x1},{y1})-({x2},{y2}).")
        return Image.fromarray(face_crop)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(src) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, (bytes, bytearray)):
            return Image.open(io.BytesIO(src)).convert("RGB")
        if hasattr(src, "read"):          # file-like object
            return Image.open(src).convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(src)}")

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        tensor = TRANSFORM(img)           # (3, 224, 224)
        return tensor.unsqueeze(0).to(self.device)   # (1, 3, 224, 224)
