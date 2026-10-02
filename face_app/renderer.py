import cv2
import numpy as np

from face_app.models import FaceDetection


class FrameRenderer:
    """Draw recognition results on OpenCV frames."""

    _KNOWN_COLOR = (0, 200, 0)
    _UNKNOWN_COLOR = (0, 165, 255)
    _TEXT_COLOR = (0, 0, 0)
    _INFO_COLOR = (255, 255, 255)

    def draw(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
    ) -> None:
        for detection in detections:
            self._draw_detection(frame, detection)

        self._draw_help(frame)

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> None:
        x1 = detection.x
        y1 = detection.y
        x2 = detection.x + detection.width
        y2 = detection.y + detection.height

        color = (
            self._KNOWN_COLOR
            if detection.is_known
            else self._UNKNOWN_COLOR
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = detection.name
        if detection.distance is not None:
            label = f"{label}  {detection.distance:.2f}"

        self._draw_label(
            frame=frame,
            text=label,
            x=x1,
            y=y1,
            background_color=color,
        )

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        background_color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        padding = 6

        (text_width, text_height), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        label_bottom = max(text_height + padding * 2, y)
        label_top = label_bottom - text_height - padding * 2

        cv2.rectangle(
            frame,
            (x, label_top),
            (x + text_width + padding * 2, label_bottom),
            background_color,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + padding, label_bottom - padding),
            font,
            font_scale,
            self._TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_help(self, frame: np.ndarray) -> None:
        cv2.putText(
            frame,
            "Q = Quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self._INFO_COLOR,
            2,
            cv2.LINE_AA,
        )
