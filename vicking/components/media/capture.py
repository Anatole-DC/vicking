from dataclasses import dataclass
from typing import Any

from river import Component, Node
import cv2 as opencv
from numpy import ndarray

from vicking.data.frame_data import FrameData


@dataclass
class CaptureComponent(Component):
    channel: int | str = 0

    def __post_init__(self):
        super().__post_init__()
        self.nodes = {"frame": Node()}
        self.camera = opencv.VideoCapture(self.channel)

    def read(self) -> ndarray:
        status, frame = self.camera.read()
        if status is None:
            return
        return frame
    
    @property
    def fps(self) -> int:
        return int(self.camera.get(opencv.CAP_PROP_FPS))

    def __call__(self) -> Any:
        image: ndarray = self.read()

        if image is None:
            return

        self.nodes["frame"].set(FrameData(self.iterations, image))

        self.iterations += 1
