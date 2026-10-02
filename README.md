# Small CPU Face Identification Project

Purpose: identify known people when they appear in a webcam.

Pipeline:

`Webcam -> YOLOv8-Face -> face crop -> DeepFace/FaceNet512 -> cosine comparison -> name/Unknown`

A GPU is not required. The sample explicitly runs YOLO on CPU.

## 1. Requirements

Recommended: Python 3.10 or 3.11.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

DeepFace/TensorFlow compatibility can vary by Python/platform version. If installation fails, use Python 3.10/3.11 in a clean virtual environment.

## 2. Add the YOLOv8-Face model

Place a YOLOv8-Face weights file here:

```text
models/yolov8n-face.pt
```

The project intentionally does not bundle third-party model weights.

## 3. Add known people

For one reference image per person:

```text
known_faces/
├── Alice.jpg
├── Bob.jpg
└── John.jpg
```

The filename becomes the displayed name.

For multiple reference images per person (recommended):

```text
known_faces/
├── Alice/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
└── Bob/
    ├── 1.jpg
    └── 2.jpg
```

The folder name becomes the displayed name.

Use clear, front-facing reference photos where the face is reasonably large.

## 4. Run

```bash
python main.py
```

Press `Q` to quit.

## Configuration

At the top of `main.py` you can change:

- `MATCH_THRESHOLD` - lower is stricter.
- `YOLO_CONFIDENCE` - face detection confidence.
- `RECOGNIZE_EVERY_N_FRAMES` - larger values reduce CPU usage.
- `CAMERA_INDEX` - use 1, 2, etc. if camera 0 is not the desired webcam.

The default match threshold is only a starting point. Tune it with your own camera, lighting, and reference photos before relying on identification results.

## Privacy

Face embeddings are biometric data. Keep reference images/embeddings protected and use the system with appropriate consent and access controls.
