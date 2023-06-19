from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import cv2 as opencv

from numpy import ndarray

from .capture import CaptureComponent


@dataclass
class VideoComponent(CaptureComponent):
    path: Path
    interval: Tuple[int, int] = (None, None)
    channel: str | int = field(init=False)

    def __post_init__(self):
        if not self.path.exists():
            raise ValueError(f"Video path {self.path} does not exist.")
        if not self.path.is_file():
            raise ValueError(f"Video path {self.path} is not a file.")
        self.channel = str(self.path.absolute())
        super().__post_init__()

        if self.interval[0] is not None:
            self.camera.set(opencv.CAP_PROP_POS_FRAMES, self.interval[0])

    @property
    def total_frame(self) -> int:
        return int(self.camera.get(opencv.CAP_PROP_FRAME_COUNT))
    
    def get_frame_at_index(self, index: int) -> ndarray | None:
        cache_pos: int = self.camera.get(opencv.CAP_PROP_POS_FRAMES)
        self.camera.set(opencv.CAP_PROP_POS_FRAMES, index)
        frame: ndarray = self.read()
        self.camera.set(opencv.CAP_PROP_POS_FRAMES, cache_pos)

        if frame is None: return

        return frame