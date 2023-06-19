from dataclasses import dataclass
from pathlib import Path

import cv2 as opencv
from river import Component, Node

from vicking.data.frame_data import FrameData


@dataclass
class ImageComponent(Component):
    image_path: Path

    def __post_init__(self):
        super().__post_init__()
        if not self.image_path.exists():
            raise ValueError(f"Path '{self.image_path.absolute()}' does not exist.")

        if not self.image_path.is_file():
            raise ValueError(f"Path '{self.image_path.absolute()}' is not a file path.")

        self.image: FrameData = FrameData(0, opencv.imread(str(self.image_path.absolute())))

        self.nodes["image"] = Node()

    def __call__(self) -> None:
        self.nodes["image"].set(self.image)
