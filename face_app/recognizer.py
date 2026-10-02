from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Important:
# Configure DeepFace before importing DeepFace itself.
from face_app import backend as _backend  # noqa: F401
from deepface import DeepFace

from face_app.config import AppConfig
from face_app.models import FaceDetection, KnownFace


LOGGER = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Detect faces with a local YOLOv8-Face model and identify them
    using FaceNet512 embeddings through DeepFace.

    Responsibilities:
        1. Load the local YOLO face detector.
        2. Build embeddings for registered people.
        3. Detect faces in webcam frames.
        4. Build embeddings for detected faces.
        5. Compare embeddings using cosine distance.
    """

    UNKNOWN_NAME = "Unknown"

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Reference embeddings loaded from known_faces/.
        self._known_faces: list[KnownFace] = []

        # Load the local YOLO face model only once.
        self._detector = self._load_detector()

        # DeepFace caches this client and reuses it in represent(). Move both
        # the model and its input-device field, so CPU mode is respected even
        # on a machine with CUDA and both pipeline stages use the same device.
        self._embedding_model = DeepFace.build_model(self._config.face_model)
        self._embedding_model.device = torch.device(self._config.yolo_device)
        self._embedding_model.model.to(self._embedding_model.device)
        self._embedding_model.model.eval()
        LOGGER.info("%s embedding device: %s", self._config.face_model, self._embedding_model.device)

    # =========================================================
    # Public API
    # =========================================================

    @property
    def known_face_count(self) -> int:
        """Number of usable reference face embeddings."""
        return len(self._known_faces)

    def load_known_faces(self) -> None:
        """
        Load reference images and create embeddings.

        Recommended directory layout:

            known_faces/
                Phee Chouk/
                    1.jpg
                    2.jpg

                Phee Pong/
                    1.jpg
                    2.jpg

                Phee Thein/
                    1.jpg
                    2.jpg

        Every image inside one person's directory is registered
        under the same person name.
        """

        directory = self._config.known_faces_dir

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in self._config.supported_image_extensions
        )

        self._known_faces.clear()

        if not image_paths:
            LOGGER.warning(
                "No reference images found in %s.",
                directory,
            )
            return

        LOGGER.info(
            "Loading %d reference image(s)...",
            len(image_paths),
        )

        for image_path in image_paths:

            person_name = self._person_name_from_path(
                image_path
            )

            try:

                embedding = (
                    self._embedding_from_registered_image(
                        image_path
                    )
                )

            except Exception:

                LOGGER.exception(
                    "Could not register face from %s",
                    image_path,
                )

                continue

            self._known_faces.append(
                KnownFace(
                    name=person_name,
                    embedding=embedding,
                )
            )

            LOGGER.info(
                "Registered %-20s <- %s",
                person_name,
                image_path.name,
            )

        LOGGER.info(
            "Loaded %d usable reference embedding(s).",
            len(self._known_faces),
        )

    def detect_and_identify(
        self,
        frame: np.ndarray,
    ) -> list[FaceDetection]:
        """
        Detect and identify every face in a webcam frame.

        Pipeline:

            OpenCV frame
                ↓
            YOLOv8-Face
                ↓
            face crop
                ↓
            FaceNet512
                ↓
            embedding
                ↓
            cosine comparison
        """

        detections: list[FaceDetection] = []

        yolo_results = self._detect_faces(frame)

        for result in yolo_results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1, y1, x2, y2 = coordinates

                confidence = float(
                    box.conf[0].cpu().item()
                )

                face_image, area = self._crop_face(
                    frame=frame,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )

                if face_image is None:
                    continue

                name, distance = self._identify_face(
                    face_image
                )

                area_x1, area_y1, area_x2, area_y2 = area

                detections.append(
                    FaceDetection(
                        name=name,
                        distance=distance,
                        confidence=confidence,
                        x=area_x1,
                        y=area_y1,
                        width=area_x2 - area_x1,
                        height=area_y2 - area_y1,
                    )
                )

        return detections

    # =========================================================
    # YOLO
    # =========================================================

    def _load_detector(self) -> YOLO:
        """Load the local YOLOv8 face model."""

        model_path = self._config.yolo_model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO face model not found: {model_path}\n"
                f"Place yolov8n-face.pt inside:\n"
                f"{model_path.parent}/"
            )

        LOGGER.info(
            "Loading YOLO face detector: %s",
            model_path,
        )

        detector = YOLO(
            str(model_path)
        )

        LOGGER.info(
            "YOLO face detector loaded."
        )

        return detector

    def _detect_faces(
        self,
        image: np.ndarray,
    ):
        """
        Run YOLO face detection.

        This method is shared by both:
            - webcam frames
            - known/reference images
        """

        try:

            return self._detector.predict(
                source=image,
                conf=self._config.yolo_confidence,
                imgsz=self._config.yolo_image_size,
                device=self._config.yolo_device,
                verbose=False,
            )

        except Exception:

            LOGGER.exception(
                "YOLO face detection failed."
            )

            return []

    # =========================================================
    # Reference Images
    # =========================================================

    def _embedding_from_registered_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        """
        Detect the main face in a registered image and create
        its FaceNet512 embedding.

        If more than one face appears in the reference image,
        the largest detected face is used.
        """

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        results = self._detect_faces(image)

        detected_boxes: list[
            tuple[int, int, int, int]
        ] = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                x1, y1, x2, y2 = coordinates

                detected_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

        if not detected_boxes:
            raise ValueError(
                f"No face detected in {image_path}"
            )

        # Reference images should normally contain one person.
        # If multiple faces exist, use the largest one.
        largest_box = max(
            detected_boxes,
            key=self._box_area,
        )

        face_image, _ = self._crop_face(
            frame=image,
            x1=largest_box[0],
            y1=largest_box[1],
            x2=largest_box[2],
            y2=largest_box[3],
        )

        if face_image is None:
            raise ValueError(
                f"Could not crop face from {image_path}"
            )

        return self._create_embedding(
            face_image
        )

    # =========================================================
    # Recognition
    # =========================================================

    def _identify_face(
        self,
        face_image: np.ndarray,
    ) -> tuple[str, float | None]:
        """
        Identify one detected face against registered people.
        """

        if not self._known_faces:
            return (
                self.UNKNOWN_NAME,
                None,
            )

        try:

            current_embedding = (
                self._create_embedding(
                    face_image
                )
            )

        except Exception:

            LOGGER.exception(
                "Could not create embedding "
                "for detected face."
            )

            return (
                self.UNKNOWN_NAME,
                None,
            )

        best_match: KnownFace | None = None

        best_distance = float("inf")

        for known_face in self._known_faces:
            distance = self._cosine_distance(
                current_embedding,
                known_face.embedding,
            )
            if distance < best_distance:
                best_distance = distance
                best_match = known_face

        if best_match is None:
            return (
                self.UNKNOWN_NAME,
                best_distance,
            )
        if (
            best_distance
            > self._config.match_threshold
        ):
            return (
                self.UNKNOWN_NAME,
                best_distance,
            )
        return (
            best_match.name,
            best_distance,
        )

    def _create_embedding(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        Create a FaceNet512 embedding from an already-cropped face.

        Detection is skipped because YOLO has already done it.
        """

        results = DeepFace.represent(
            img_path=face_image,
            model_name=self._config.face_model,

            # YOLO already detected the face.
            detector_backend="skip",

            enforce_detection=False,

            # We are providing the cropped face directly.
            align=False,
        )

        if not results:
            raise ValueError(
                "DeepFace returned no embedding."
            )

        return np.asarray(
            results[0]["embedding"],
            dtype=np.float32,
        )

    # =========================================================
    # Face Cropping
    # =========================================================

    def _crop_face(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[
        np.ndarray | None,
        tuple[int, int, int, int],
    ]:
        """
        Crop a detected face with a small margin.

        A margin is useful because extremely tight crops can remove
        forehead/chin information that helps face recognition.
        """

        frame_height, frame_width = frame.shape[:2]

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return None, (
                x1,
                y1,
                x2,
                y2,
            )

        margin_x = int(
            width
            * self._config.face_margin_ratio
        )

        margin_y = int(
            height
            * self._config.face_margin_ratio
        )

        crop_x1 = max(
            0,
            x1 - margin_x,
        )

        crop_y1 = max(
            0,
            y1 - margin_y,
        )

        crop_x2 = min(
            frame_width,
            x2 + margin_x,
        )

        crop_y2 = min(
            frame_height,
            y2 + margin_y,
        )

        face_image = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if face_image.size == 0:
            return None, (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2,
            )

        return face_image, (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )

    # =========================================================
    # Utilities
    # =========================================================

    def _person_name_from_path(
        self,
        image_path: Path,
    ) -> str:
        """
        Determine the person's name from the directory structure.

        Example:

            known_faces/Phee Chouk/1.jpg

        becomes:

            Phee Chouk
        """

        root = self._config.known_faces_dir

        if image_path.parent == root:

            raw_name = image_path.stem

        else:

            raw_name = image_path.parent.name

        return (
            raw_name
            .replace("_", " ")
            .strip()
        )

    @staticmethod
    def _box_area(
        box: tuple[
            int,
            int,
            int,
            int,
        ],
    ) -> int:
        """Calculate bounding-box area."""

        x1, y1, x2, y2 = box

        return max(
            0,
            x2 - x1,
        ) * max(
            0,
            y2 - y1,
        )

    @staticmethod
    def _cosine_distance(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Return cosine distance.

        0.0:
            very similar

        Larger value:
            increasingly different
        """

        denominator = float(
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator == 0.0:
            return 1.0

        similarity = float(
            np.dot(a, b)
            / denominator
        )

        return 1.0 - similarity
