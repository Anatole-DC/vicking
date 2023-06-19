from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2 as opencv
from numpy import ndarray

from .detector import DetectorComponent


@dataclass
class HaarcascadeDetectorComponent(DetectorComponent):
    model: Path

    def __post_init__(self):
        super().__post_init__()
        if not self.model.exists():
            raise FileNotFoundError(f"Wrong model path '{self.model.absolute()}'")
        if not self.model.is_file():
            raise ValueError(f"Haarcascade detector model must be a file")
        self.detector = opencv.CascadeClassifier(str(self.model.absolute()))

    def detect(self, frame: ndarray) -> List:
        return [
            detection
            for detection in self.detector.detectMultiScale(
                frame,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(30, 30),
                flags=opencv.CASCADE_SCALE_IMAGE,
            )
        ]
