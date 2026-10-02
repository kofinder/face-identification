from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Central application configuration."""

    known_faces_dir: Path = Path("known_faces")

    # ---------------------------------------------------------
    # Local YOLO face detector
    # ---------------------------------------------------------

    yolo_model_path: Path = Path("models/yolov8n-face.pt")

    yolo_confidence: float = 0.50
    yolo_image_size: int = 640
    yolo_device: str = "cpu"

    # Adds a little surrounding area around the detected face.
    face_margin_ratio: float = 0.10

    # ---------------------------------------------------------
    # Camera
    # ---------------------------------------------------------

    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # ---------------------------------------------------------
    # Face recognition
    # ---------------------------------------------------------

    face_model: str = "Facenet512"

    match_threshold: float = 0.30

    # Run recognition every N webcam frames.
    process_every_n_frames: int = 5

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    window_title: str = "Face Identification"

    supported_image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )