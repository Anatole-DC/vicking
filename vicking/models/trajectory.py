from dataclasses import dataclass, field
from typing import List
from datetime import datetime, timedelta
import uuid

import cv2 as opencv
from cvermeer.drawing import Drawable
from cvermeer.colors import Color
from numpy import ndarray

@dataclass
class Trajectory(Drawable):
    predictions: List = field(default_factory=list)
    last_update: datetime = field(init=False, default_factory=datetime.now)
    uuid: int = field(init=False)
    color: Color = field(default_factory=Color.random_color)


    def __post_init__(self):
        self.uuid = uuid.uuid4().int

    def last(self):
        return self.predictions[-1]

    def update(self, value):
        self.predictions.append(value)
        self.last_update = datetime.now()

    def is_lost(
        self, time_to_trajectory_loss: timedelta = timedelta(seconds=2)
    ) -> bool:
        return datetime.now() - self.last_update >= time_to_trajectory_loss

    def draw(self, frame: ndarray, color: Color = None) -> None:
        [
            opencv.line(frame, pt1.tuple(), pt2.tuple(), self.color.cv_color, 2)
            for pt1, pt2 in zip(self.predictions, self.predictions[1:])
        ]