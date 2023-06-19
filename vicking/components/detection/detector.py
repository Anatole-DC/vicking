from dataclasses import dataclass, field
from typing import Any, Dict, List
from copy import deepcopy

from river import Component, Node, Data
from river.lib.pipes.buffer import Buffer, BufferStorageMode
from cvermeer.shapes import BBox
from numpy import ndarray
import cv2 as opencv

from vicking.data.frame_data import FrameData


@dataclass
class CustomNode(Node):
    def __post_init__(self):
        self.buffer = Buffer(BufferStorageMode.BUFF)

@dataclass
class DetectionData(Data):
    detections: List[BBox]

    @property
    def data(self):
        return self.detections

@dataclass
class DetectorComponent(Component):
    current_token: int = field(init=False, default=-1)

    def __post_init__(self):
        super().__post_init__()
        self.nodes = {"frame": Node(), "detection_frame": Node(), "detections": Node()}

    def detect(self, frame: ndarray) -> List[BBox]:
        ...

    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        data: FrameData = self.nodes["frame"].get()
        if data is None:
            return

        self.current_token = data.token

        frame: ndarray = data.image
        detections: List = self.detect(frame)

        self.nodes["detections"].set(DetectionData(data.token, detections))

        self.iterations += 1
