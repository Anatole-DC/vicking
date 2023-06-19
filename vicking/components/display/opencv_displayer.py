from dataclasses import dataclass, field
from typing import Any, Dict, List

from river import Component, Node
import cv2 as opencv
from numpy import ndarray

from vicking.data.frame_data import FrameData


@dataclass
class OpenCVDisplayerComponent(Component):
    label: str = None
    last_frame: ndarray = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.nodes = {"frame": Node()}
        self.window_name: str = self.label if self.label is not None else self.name
        opencv.namedWindow(self.window_name, opencv.WINDOW_NORMAL)

    def __call__(self) -> Any:
        frame: FrameData = self.nodes["frame"].get()

        if frame is None:
            return

        opencv.imshow(self.window_name, frame.image)
        opencv.waitKey(1)
