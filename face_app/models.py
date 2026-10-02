from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KnownFace:
    """A registered person and one reference face embedding."""

    name: str
    embedding: np.ndarray


@dataclass(slots=True)
class FaceDetection:
    """Recognition result for one face in a webcam frame."""

    name: str
    distance: float | None
    confidence: float
    x: int
    y: int
    width: int
    height: int

    @property
    def is_known(self) -> bool:
        return self.name != "Unknown"
