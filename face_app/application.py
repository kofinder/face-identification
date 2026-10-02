import logging

import cv2

from face_app.config import AppConfig
from face_app.models import FaceDetection
from face_app.recognizer import FaceRecognizer
from face_app.renderer import FrameRenderer

LOGGER = logging.getLogger(__name__)


class FaceIdentificationApp:
    """Coordinate registration, webcam capture, recognition, and rendering."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._recognizer = FaceRecognizer(config)
        self._renderer = FrameRenderer()

    def run(self) -> None:
        """Start the face identification application."""
        LOGGER.info("Starting %s", self._config.window_title)

        self._recognizer.load_known_faces()
        if self._recognizer.known_face_count == 0:
            LOGGER.error(
                "No usable known faces. Add images under '%s' and restart.",
                self._config.known_faces_dir,
            )
            return

        camera = self._open_camera()

        frame_count = 0
        current_detections: list[FaceDetection] = []

        try:
            LOGGER.info("Webcam started. Press Q to quit.")

            while True:
                success, frame = camera.read()
                if not success:
                    LOGGER.error("Could not read a frame from the webcam.")
                    break

                frame_count += 1

                # Face recognition is the expensive part. Run it periodically,
                # but continue drawing the most recent result between passes.
                if (
                    frame_count == 1
                    or frame_count % self._config.process_every_n_frames == 0
                ):
                    current_detections = (
                        self._recognizer.detect_and_identify(frame)
                    )

                self._renderer.draw(frame, current_detections)

                cv2.imshow(self._config.window_title, frame)

                if self._quit_requested():
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()
            LOGGER.info("Application stopped.")

    def _open_camera(self) -> cv2.VideoCapture:
        camera = cv2.VideoCapture(self._config.camera_index)

        if not camera.isOpened():
            camera.release()
            raise RuntimeError(
                f"Cannot open webcam at index "
                f"{self._config.camera_index}."
            )

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self._config.camera_width,
        )
        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self._config.camera_height,
        )

        return camera

    @staticmethod
    def _quit_requested() -> bool:
        return (cv2.waitKey(1) & 0xFF) == ord("q")
