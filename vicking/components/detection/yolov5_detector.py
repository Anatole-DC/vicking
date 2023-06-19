from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from cvermeer.shapes import BBox, Point
import numpy

from .detector import DetectorComponent

@dataclass
class YOLOV5DetectorComponent(DetectorComponent):
    model_path: Path
    device: torch.device = torch.device('cuda')

    def __post_init__(self):
        super().__post_init__()
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            torch.cuda.empty_cache()
            self.device = self.device
        self.detector = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=self.model_path.absolute(),
            force_reload=True
        )
        self.names = self.detector.names
        print(self.device)
        self.detector.to(self.device)

    def detect(self, frame: numpy.ndarray) -> List[BBox]:
        detections: List[BBox] = []
        predictions = self.detector([frame])
        cord = predictions.xyxyn[0][:, :-1]

        for result in cord:
            x1, y1, x2, y2, _ = result
            bbox = BBox(
                Point(float(x1), float(y1)).to_scale(frame),
                Point(float(x2), float(y2)).to_scale(frame)
            )
            detections.append(bbox)

        return detections
