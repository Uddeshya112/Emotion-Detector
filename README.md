# EmotiSense AI — Emotion Detector

A web app that predicts one of seven facial-expression classes from an uploaded image or a live webcam frame.

> **Live app:** [https://emotion-detector-0tuc.onrender.com/](https://emotion-detector-0tuc.onrender.com/)  
> **Health check:** [https://emotion-detector-0tuc.onrender.com/health](https://emotion-detector-0tuc.onrender.com/health)

The live app is hosted on Render and connected to this GitHub repository. The current Render free instance may sleep when idle, so its first request after inactivity can take longer to respond.

## Features

- Upload an image and see the predicted expression and confidence scores.
- Use a webcam for repeated predictions while the camera is enabled.
- View the seven class scores for each prediction.
- Review recent results in the browser.
- Switch between light and dark appearance.
- Use the health endpoint to check whether the server and model are ready.

This is a facial-expression classifier. Its output is a model prediction about visible expression, not a reliable measure of a person's internal emotional state.

## Emotion classes

| Label | Emoji |
| --- | --- |
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprise | 😲 |

## How it works

1. The browser sends an image to the Flask server.
2. OpenCV attempts to find the largest face with a Haar cascade. If it cannot find a face, the app processes the full image.
3. The image is converted to grayscale, resized to 48 × 48, expanded to three channels, and normalized.
4. A PyTorch ResNet18 classifier produces scores for the seven classes.
5. The server returns the top class and scores as JSON.

The deployed predictor loads `best_emotion_model.pth`. That checkpoint is required for meaningful predictions. If it is missing, the predictor can fall back to randomly initialized weights in demo mode, which should not be treated as real predictions.

## Run locally

### Requirements

- Python 3.11
- pip
- A webcam for the optional camera feature
- A modern browser; camera access requires permission and a secure context (localhost is supported)

### Install and start

**Windows PowerShell**

```powershell
git clone https://github.com/Uddeshya112/Emotion-Detector.git
cd Emotion-Detector
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

**macOS / Linux**

```bash
git clone https://github.com/Uddeshya112/Emotion-Detector.git
cd Emotion-Detector
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The application selects CUDA when PyTorch detects a compatible GPU; otherwise it runs on CPU. The Render deployment uses the repository's Docker configuration and is not dependent on a GPU.

## HTTP API

The JSON endpoints return the predicted class, confidence, emoji, and a score for every class.

### Health

```http
GET /health
```

Example:

```json
{
  "status": "ok",
  "model_ready": true
}
```

### Predict from an uploaded image

```http
POST /predict
Content-Type: multipart/form-data
```

Send the image in a multipart field named `image`. Supported extensions are PNG, JPG/JPEG, GIF, BMP, and WebP. The server limits request bodies to 16 MiB and verifies that the upload is a readable image.

```bash
curl -X POST "http://127.0.0.1:5000/predict" \
  -F "image=@photo.jpg"
```

Example success response:

```json
{
  "success": true,
  "emotion": "Happy",
  "confidence": 0.91,
  "emoji": "😊",
  "all_scores": {
    "Angry": 0.01,
    "Disgust": 0.0,
    "Fear": 0.01,
    "Happy": 0.91,
    "Neutral": 0.03,
    "Sad": 0.01,
    "Surprise": 0.03
  }
}
```

### Predict from a webcam frame

```http
POST /webcam
Content-Type: application/json
```

Send JSON with an `image` property containing a base64-encoded image. A data URL prefix such as `data:image/jpeg;base64,` is accepted.

```json
{
  "image": "data:image/jpeg;base64,<base64-image-data>"
}
```

Errors are returned as JSON with `success: false` and an `error` message.

## Model and training files

The runtime model is a ResNet18 checkpoint at `best_emotion_model.pth`. The repository also contains experimental training scripts and other model artifacts:

- `train_pytorch_fast.py` trains a PyTorch ResNet18 using the expected 48 × 48 input pipeline and saves `best_emotion_model.pth`.
- `train_pytorch_resnet.py` is an alternative ResNet50 experiment. Its architecture does not match the deployed ResNet18 predictor, so its checkpoint should not replace the runtime checkpoint without adapting the predictor.
- `train.py` and `train_resnet.py` are TensorFlow/Keras experiments. Their `.h5` / `.keras` outputs are not loaded by the Flask app.
- `check_dataset.py` counts images in the dataset folders.
- `gpu_test.py` reports the local PyTorch and CUDA availability.

The training scripts expect a dataset under `archive/` with training and testing image folders organized by class. Training dependencies and dataset setup are separate from the web app's `requirements.txt`; the TensorFlow experiments need TensorFlow installed separately. The Docker build excludes `archive/`, so the dataset is not sent to the Render service.

## Deploy on Render

The production app is currently served from Render at the live URL above.

To deploy your own instance:

1. Fork or connect this repository in Render.
2. Create a **Web Service** using the Docker runtime, or use the included `render.yaml` Blueprint.
3. Set the repository root as the service root directory.
4. Deploy. The Dockerfile installs the Python dependencies and starts Gunicorn on Render's `PORT`.
5. Use `/health` as the health-check path.

The service runs one Gunicorn worker with four threads and a 120-second request timeout. On a free instance, idle spin-down can make the first request slow.

## Repository layout

```text
.
├── app.py                    # Flask routes and application startup
├── predictor.py              # Face preprocessing and ResNet18 inference
├── best_emotion_model.pth    # Checkpoint used by the deployed predictor
├── emotion_model.h5          # Separate Keras experiment artifact
├── templates/
│   └── index.html            # Main web interface
├── static/
│   ├── script.js             # Upload, webcam, and browser-side behavior
│   └── style.css             # Interface styles
├── archive/                  # Training/testing image dataset
├── train_pytorch_fast.py     # PyTorch ResNet18 training experiment
├── train_pytorch_resnet.py   # PyTorch ResNet50 experiment
├── train.py                  # Keras CNN experiment
├── train_resnet.py           # Keras ResNet experiment
├── check_dataset.py          # Dataset inspection helper
├── gpu_test.py               # PyTorch/CUDA diagnostic helper
├── requirements.txt          # Web app Python dependencies
├── Dockerfile                # Container configuration used by Render
├── render.yaml               # Render Blueprint configuration
└── .dockerignore             # Files excluded from the Docker build
```

## Main dependencies

- Flask for the web server
- PyTorch and Torchvision for inference
- OpenCV headless for face detection and image processing
- Pillow for image validation and decoding

See [`requirements.txt`](requirements.txt) for pinned versions.

## License

No license file is currently included. Unless a license is added to the repository, reuse and redistribution are not explicitly granted.
