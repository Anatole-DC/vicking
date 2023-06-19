from dataclasses import dataclass
from typing import List

import cv2 as opencv
import numpy
from river import Component, Node

from vicking.data.frame_data import FrameData

@dataclass
class FrameBuilderComponent(Component):
    """FramebuilderComponent takes frames and stack them together to form one frame"""

    def __post_init__(self):
        super().__post_init__()
        self.nodes["frame1"] = Node()
        self.nodes["frame2"] = Node()
        self.nodes["output_frame"] = Node()

    def __call__(self):
        frame1: FrameData = self.nodes["frame1"].get()
        frame2: FrameData = self.nodes["frame2"].get()

        if frame1 is None or frame2 is None: return

        self.nodes["output_frame"].set(FrameData(frame1.token, self.stack_frames([frame1.image, frame2.image])))


    def stack_frames(self, frames: List[numpy.ndarray]) -> numpy.ndarray:
        # Get dimensions of input frames
        max_height, max_width, max_channels = max((frame.shape for frame in frames), key=lambda x: x[0]*x[1])

        # Create empty output frame with double the width
        output_frame = numpy.zeros((max_height, max_width * len(frames), max_channels), dtype=numpy.uint8)

        # Resize and stack input frames horizontally on the output frame
        for i, frame in enumerate(frames):
            resized_frame = opencv.resize(frame, (max_width, max_height))
            output_frame[:, i * max_width : (i+1) * max_width, :] = resized_frame

        return output_frame
