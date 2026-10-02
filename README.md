# Face Identification — CPU and NVIDIA GPU

Identify registered people in a webcam stream with this pipeline:

`Webcam → YOLOv8-Face → face crop → DeepFace / FaceNet512 → cosine comparison → name or Unknown`

Both AI models use **PyTorch**. CPU is the default; NVIDIA CUDA acceleration is optional. Camera capture, cropping, drawing, and cosine comparison still run on CPU in GPU mode.

## 1. Set up the project

Use Python **3.10 or 3.11**, a webcam, and a desktop session where OpenCV can display a window. Run commands from the project directory so relative model and image paths work.

### Ubuntu / Linux

```bash
git clone https://github.com/kofinder/face-identification.git
cd face-identification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `venv` is unavailable on Ubuntu:

```bash
sudo apt install python3-venv
```

### Windows PowerShell

```powershell
git clone https://github.com/kofinder/face-identification.git
cd face-identification
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Choose **one** installation below.

## 2A. Install for CPU

For Linux or Windows:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

For macOS, install `torch torchvision` using the command from the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install `requirements.txt`. This app's GPU option targets NVIDIA CUDA; Apple MPS and AMD ROCm are not supported by this option.

Check CPU mode without downloading FaceNet weights or opening the webcam:

```bash
python main.py --device cpu --check-device
```

Expected log: `YOLO and FaceNet device: CPU`.

## 2B. Install for NVIDIA GPU

You need an NVIDIA CUDA-capable GPU, a compatible NVIDIA driver, and a CUDA-enabled PyTorch build. A CPU-only PyTorch build cannot use your GPU. Use the [official PyTorch installer](https://pytorch.org/get-started/locally/) to select a CUDA build compatible with your driver and GPU architecture; do not choose a CUDA version solely by copying someone else's command.

First check the driver:

```bash
nvidia-smi
```

In the activated virtual environment, run the install command supplied by the PyTorch installer for **torch and torchvision**, then:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python main.py --device gpu --check-device
```

`CUDA available` must be `True`. The app also executes a small CUDA operation to catch driver/build/architecture problems before loading its models. Expected log: `YOLO and FaceNet device: cuda:0 (...)`.

If you previously installed CPU-only PyTorch, uninstall `torch` and `torchvision`, then install the CUDA versions using the official command:

```bash
python -m pip uninstall -y torch torchvision
# Run the torch/torchvision CUDA install command from the official installer here.
python -m pip install -r requirements.txt
```

The same GPU-enabled environment can also run `--device cpu`; you do not need to reinstall dependencies each time. Standard PyTorch wheels include the CUDA runtime dependencies; you still need the NVIDIA driver. A separate system CUDA toolkit is normally unnecessary for this project's prebuilt wheels.

**Platform notes:** GPU mode uses PyTorch for both models, including on native Windows; TensorFlow GPU setup is not needed. Jetson devices require a PyTorch build matched to JetPack rather than these desktop installation commands. Webcam access must be configured separately if using WSL, containers, or remote sessions.

## 3. Check the face detector weights

The repository currently includes:

```text
models/yolov8n-face.pt
```

If your checkout lacks this file, put compatible **YOLOv8 face-detection weights** at that path. Standard object-detection `yolov8n.pt` is not a substitute. Model loading and inference errors should be checked in the startup logs.

DeepFace downloads its FaceNet512 PyTorch weights on the first run, normally under `~/.deepface/weights/`. That first run needs internet access and can take longer. Later runs reuse the weights.

## 4. Register known people

Replace or extend the sample photos in `known_faces/`. Use a folder for each person:

```text
known_faces/
├── Alice/
│   ├── 1.jpg
│   └── 2.jpg
└── Bob/
    ├── 1.jpg
    └── 2.jpg
```

The folder name becomes the displayed name. A top-level file such as `known_faces/Alice.jpg` also works; its filename becomes the name. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.

Use clear photos with one face per image and include different lighting/angles. When a reference photo contains multiple faces, the largest detected face is registered. Reference embeddings are rebuilt in memory at startup.

## 5. Run

| Command | YOLO detection | FaceNet512 recognition | Behavior |
| --- | --- | --- | --- |
| `python main.py` | CPU | CPU | Default |
| `python main.py --device cpu` | CPU | CPU | Explicit CPU mode |
| `python main.py --device gpu` | CUDA GPU 0 | CUDA GPU 0 | Exit with an error if CUDA cannot pass the startup check |
| `python main.py --device auto` | GPU when usable, otherwise CPU | Same selected device | Select at startup and log the result |

Press **Q** in the webcam window to quit.

Additional options:

```bash
python main.py --device cpu --camera 1
python main.py --device cpu --every-n-frames 10
python main.py --device gpu --every-n-frames 1
python main.py --help
```

`--every-n-frames` controls how often detection and recognition run. Between passes the app draws the previous result, so larger values reduce compute usage but can leave labels/boxes behind moving faces. A GPU may improve inference speed; the gain depends on hardware, face count, and camera speed. Choose the frame interval after testing your camera.

To select another physical GPU on Linux, expose only that GPU before launch; the app uses the first visible GPU as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --device gpu
```

Device selection happens at startup. Restart the process to change modes. Auto mode checks CUDA at startup; it does not recover automatically from later inference errors or GPU out-of-memory errors.

## Configuration

Defaults live in **`face_app/config.py`**, inside `AppConfig`:

| Setting | Default | Purpose |
| --- | --- | --- |
| `yolo_device` | `"cpu"` | CLI default: `cpu`, `gpu`, or `auto`; resolved to `cpu` or `cuda:0` for both models |
| `yolo_model_path` | `models/yolov8n-face.pt` | Local face detector weights |
| `known_faces_dir` | `known_faces` | Reference images |
| `yolo_confidence` | `0.50` | Minimum detection confidence |
| `yolo_image_size` | `640` | Detector input size |
| `face_model` | `"Facenet512"` | DeepFace recognition model |
| `match_threshold` | `0.30` | Maximum cosine distance to accept a match; lower is stricter |
| `process_every_n_frames` | `5` | Detection/recognition interval |
| `camera_index` | `0` | Webcam index |
| `camera_width`, `camera_height` | `1280`, `720` | Requested resolution; the camera may negotiate another size |

The application selects DeepFace's PyTorch backend before importing DeepFace. `requirements.txt` pins DeepFace to `0.0.101`, whose PyTorch FaceNet client supports the device assignment used here. That release can still install TensorFlow-related packages as dependencies; this app runs recognition on PyTorch.

Tune the match threshold using your own photos, camera, and lighting. Recheck recognition accuracy if you change the model or backend.

## Troubleshooting

- **`GPU requested, but PyTorch cannot use CUDA`:** run `nvidia-smi`, check `torch.version.cuda` and `torch.cuda.is_available()` inside the activated environment, and reinstall the appropriate PyTorch build. `torch.version.cuda` being `None` means a CPU-only build. Use `--device cpu` to keep working.
- **CUDA kernel / unsupported architecture error:** select a PyTorch build that supports your GPU and driver using the official installer. Auto mode falls back to CPU if its startup CUDA probe fails.
- **CUDA out of memory:** close other GPU applications, use a smaller `yolo_image_size`, or restart with `--device cpu`. Increasing the frame interval reduces frequency rather than guaranteeing lower peak memory.
- **`libGL.so.1` or GUI library error on Ubuntu:** install `libgl1` and `libglib2.0-0` using your package manager. Use `opencv-python`, not `opencv-python-headless`, because this app opens a window.
- **Camera cannot open:** close applications using the webcam and try `--camera 1` or another index. Check camera permissions and use a local desktop session.
- **No usable known faces:** check the registration logs, photo paths, model weights, and that each photo contains a detectable face.
- **Weight download fails:** check network access and DeepFace's logs; if a cached weights file is incomplete, remove that specific file and allow it to download again.

## Tests

Device-selection and model-device wiring tests use fakes, so they do not need a GPU, webcam, or downloaded models:

```bash
python -m unittest discover -s tests -v
```

These tests do not replace testing webcam recognition on your own CPU/GPU hardware.

## Privacy

Face images and embeddings are biometric data. Protect reference photos, use appropriate consent and access controls, and remove sample identities you do not intend to recognize. This demo is not a liveness or anti-spoofing system.
