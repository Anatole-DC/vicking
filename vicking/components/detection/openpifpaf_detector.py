from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from cvermeer.colors import Color

import openpifpaf
from river import Node, Data
from cvermeer.shapes import Point, BBox
from cvermeer.shapes.posture import Posture
from cvermeer.drawing import Drawable
from numpy import ndarray
import torch

from .detector import DetectorComponent


@dataclass
class PosturesData(Data, Drawable):
    postures: List[Posture]

    @property
    def data(self):
        return self.postures

    def draw(self, frame: ndarray) -> None:
        return [posture.draw(frame) for posture in self.postures]


@dataclass
class OpenpifpafDetectorComponent(DetectorComponent):
    device: str = 'cuda'

    def __post_init__(self):
        super().__post_init__()
        self.detector: openpifpaf.Predictor = openpifpaf.Predictor(
            checkpoint="shufflenetv2k16"
        )
        self.nodes["poses"] = Node()
        self.nodes["ground_positions"] = Node()

    def detect(self, frame: ndarray) -> List[BBox]:
        predictions, _, _ = self.detector.numpy_image(frame)

        postures: List[Posture] = []
        for prediction in predictions:
            postures.append(
                Posture(
                    [
                        Point(int(keypoint[0]), int(keypoint[1]))
                        for keypoint in prediction.data.tolist()
                    ]
                )
            )

        self.nodes["poses"].set(PosturesData(self.current_token, postures))
        self.nodes["ground_positions"].set(
            PosturesData(
                self.current_token, [posture.ground_position for posture in postures]
            )
        )
        bboxes: List[BBox] = [posture.bbox for posture in postures]

        return bboxes
