from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Tuple

import cv2 as opencv

from river import Component, Node

@dataclass
class RecorderComponent(Component):
    """Records the input frame into a video file."""

    video_path: Path
    video_name: str
    resolution: Tuple[int, int] = (1080, 720)
    fps: int = 25

    def __post_init__(self):
        super().__post_init__()
        if self.video_path.exists():
            if not self.video_path.is_dir():
                raise ValueError(f"Argument 'video_path' must be a path to a folder.")
        else:
            logging.warning(f"Path '{self.video_path}' does not exist, therefore will be created")
            self.video_path.mkdir()

        self.writer: opencv.VideoWriter = opencv.VideoWriter(
            f"{self.video_path.absolute()}/{self.video_name}.avi",
            opencv.VideoWriter_fourcc(*'XVID'),
            self.fps,
            self.resolution
        )

        self.nodes["frame"] = Node()

    def __call__(self) -> None:
        frame = self.nodes["frame"].get()

        if frame is None: return

        resized_frame = opencv.resize(frame.image, self.resolution)
        self.writer.write(resized_frame)
