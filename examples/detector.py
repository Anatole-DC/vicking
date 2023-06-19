from river import Pipeline, Module, Node, ExecutionMode
from pathlib import Path

from vicking.components import (
    CaptureComponent,
    OpenCVDisplayerComponent,
    HaarcascadeDetectorComponent,
)


DetectionPipeline = Pipeline(
    name="DetectionPipeline",
    description="A simple pipeline to detect faces.",
    components={
        "camera": CaptureComponent(),
        "detection_module": Module(
            name="DetectionModule",
            nodes={
                "frame": Node(),
                "detection_frame": Node(),
            },
            components={
                "detector": HaarcascadeDetectorComponent(
                    Path("/data/models/Haarcascade/haarcascade_frontalface_default.xml")
                )
            },
            links=[
                (("nodes", "frame"), ("detector", "frame")),
                (("detector", "detection_frame"), ("nodes", "detection_frame")),
            ],
        ),
        "displayer": OpenCVDisplayerComponent(label="Display - Window"),
        "detection_displayer": OpenCVDisplayerComponent(label="Detection"),
    },
    links=[
        (("camera", "frame"), ("detection_module", "frame")),
        (("camera", "frame"), ("displayer", "frame")),
        (("detection_module", "detection_frame"), ("detection_displayer", "frame")),
    ],
)


if __name__ == "__main__":
    DetectionPipeline()
