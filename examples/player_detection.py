from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Any, Dict, List
from copy import deepcopy
from datetime import timedelta
import os

import torch
from cvermeer.colors import Color
from numpy import ndarray
from river import Pipeline, Module, Node, Data, Component
from cvermeer.shapes.posture import Posture
from cvermeer.shapes import Point, BBox
from cvermeer.drawing import Drawable
import cv2 as opencv
from river.lib.utils.plantuml_export import export_plantuml

from vicking.components import (
    VideoComponent,
    OpenpifpafDetectorComponent,
    IOUTrackerComponent,
    DataWriterComponent,
    RecorderComponent,
    DrawComponent,
    OpenCVDisplayerComponent,
    FrameBuilderComponent,
    PostureProjectionMapComponent,
    ImageComponent,
    YOLOV5DetectorComponent
)
from vicking.projects import Project
from vicking.models import Trajectory
from vicking.data.frame_data import FrameData


os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:310"



@dataclass
class PlayerData(Drawable):
    uuid: int
    posture: Posture
    position: Point
    color: Color = field(default_factory=Color.random_color)

    def draw(self, frame: ndarray) -> None:
        self.posture.draw(frame)
        self.posture.bbox.draw(frame, self.color)


@dataclass
class Players(Data):
    players: List[PlayerData]


@dataclass
class BallData(Data, Drawable):
    uid: int
    position: Point

    def draw(self, frame: ndarray) -> None:
        self.position.draw()


@dataclass
class FrameDataStructure(Data, Drawable):
    players: List[PlayerData]
    events: List[str]
    balls: List[BBox]

    def draw(self, frame: ndarray) -> None:
        [player.draw(frame) for player in self.players]
        [ball.draw(frame, Color(125, 0, 125)) for ball in self.balls]


@dataclass
class EventData(Data):
    event: List[str]

    @property
    def data(self):
        return self.event

@dataclass
class PlayerFieldProjectionAnalysisComponent(PostureProjectionMapComponent):
    def __post_init__(self):
        super().__post_init__()
        self.nodes["out_of_bounds"] = Node()
    
    def __call__(self):
        postures_data = self.nodes["points"].get()

        if postures_data is None: return

        out_of_bounds: List[str] = []
        for posture in postures_data.objects:
            if posture.ground_position is None: continue
            if not self.is_point_in_field(self.project_point(posture.ground_position)):
                out_of_bounds.append(f"{posture.uuid} is out of bounds")

        for posture in postures_data.objects:
            posture.__class__ =  make_dataclass(
                f"{posture.__class__.__name__}",
                fields=[
                    ("position", Point, field(
                default=self.project_point(posture.ground_position) if posture.ground_position is not None else None
                ))
                ],
                bases=(type(posture),),
            )
        
        self.nodes["out_of_bounds"].set(EventData(postures_data.token, out_of_bounds))
        self.nodes["projections"].set(postures_data)

@dataclass
class FrameDataStructureBuilder(Component):
    def __post_init__(self):
        super().__post_init__()
        self.nodes["postures"] = Node()
        self.nodes["events"] = Node()
        self.nodes["balls"] = Node()
        self.nodes["frame_data"] = Node()

    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        players: Data = self.nodes["postures"].get()
        events: EventData = self.nodes["events"].get()
        balls: Data = self.nodes["balls"].get()
        if players is None or events is None or balls is None: return
        self.nodes["frame_data"].set(
            FrameDataStructure(
                players.token,
                [PlayerData(
                    player.uuid,
                    Posture(player.body_keypoints),
                    player.position,
                    player.color
                ) for player in players.data],
                events.data,
                balls.data
            )
        )

@dataclass
class DrawProjections(DrawComponent):
    trajectory_buffer: Dict[int, Trajectory] = field(init=False, default_factory=dict)

    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        frame = self.nodes["frame"].get()
        datas: FrameDataStructure = self.nodes["datas"].get()

        if frame is None or datas is None:
            return
        
        # Add trajectories to the buffer
        for data in datas.players:
            if data.position is None: continue

            data.position.x = int(data.position.x)
            data.position.y = int(data.position.y)
            if data.uuid not in self.trajectory_buffer.keys():
                self.trajectory_buffer[data.uuid] = Trajectory([data.position], data.color)
            else:
                self.trajectory_buffer[data.uuid].update(data.position)

        # Draw all trajectories
        new_frame = deepcopy(frame.image)
        [trajectory.draw(new_frame) for trajectory in self.trajectory_buffer.values()]

        pop_key: List[int] = []
        for uuid, trajectory in self.trajectory_buffer.items():
            if trajectory.is_lost(timedelta(seconds=5)): pop_key.append(uuid)
        [self.trajectory_buffer.pop(uuid) for uuid in pop_key]

        self.nodes["output_frame"].set(FrameData(self.iterations, new_frame))

        self.iterations += 1


video = VideoComponent(
    Path(
        "/home/anatole/Projets/ocular/vicking_old/assets/videos/basketball.mp4"
    ),
    (3700, -1)
)
image = ImageComponent(
    Path("assets/basketball_court.jpg")
)


PlayerDetectionPipeline = Pipeline(
    name="PlayerDetectionPipeline",
    components={
        "camera": video,
        "image": image,
        "player_position_tracking": Module(
            name="PlayerPositionTracking",
            components={
                "player_detector": OpenpifpafDetectorComponent(),
                "tracker": IOUTrackerComponent(Posture, 0.2),
                "projection": PlayerFieldProjectionAnalysisComponent(
                    from_image=video.get_frame_at_index(0),
                    to_map=image.image.image,
                    points_on_image=[Point(x=0.16770833333333332, y=0.7425925925925926), Point(x=0.5859375, y=0.5777777777777777), Point(x=0.9510416666666667, y=0.6305555555555555), Point(x=0.3515625, y=0.9944444444444445)],
                    points_on_map=[Point(x=0.05223880597014925, y=0.9565217391304348), Point(x=0.048507462686567165, y=0.049689440993788817), Point(x=0.5, y=0.043478260869565216), Point(x=0.47947761194029853, y=0.906832298136646)]
                ),
            },
            nodes={"frame": Node(), "tracked_players": Node(), "events": Node()},
            links=[
                (("nodes", "frame"), ("player_detector", "frame")),
                (("player_detector", "poses"), ("tracker", "detections")),
                (("tracker", "tracked_objects"), ("projection", "points")),
                (("projection", "projections"), ("nodes", "tracked_players")),
                (("projection", "out_of_bounds"), ("nodes", "events")),
            ],
        ),
        "ball_detection_module": Module(
            name="BallDetectionModule",
            components={
                "detector": YOLOV5DetectorComponent(
                    Path("/data/models/ocular/basketball_detector_640.pt"),
                    torch.device('cuda')
                )
            },
            nodes={
                "frame": Node(),
                "balls": Node()
            },
            links=[
                (("nodes", "frame"), ("detector", "frame")),
                (("detector", "detections"), ("nodes", "balls")),
            ]
        ),
        "data_register": Module(
            name="DataRegistererModule",
            nodes={
                "tracked_players": Node(),
                "events": Node(),
                "balls": Node(),
                "data_output": Node(),
            },
            components={
                "builder": FrameDataStructureBuilder(),
                "data_writer": DataWriterComponent(
                    Project(
                        "player_detection",
                        Path("db"),
                        Path(
                            "/home/anatole/Projets/ocular/vicking_old/assets/videos/basketball.mp4"
                        ),
                    )
                ),
            },
            links=[
                (("nodes", "tracked_players"), ("builder", "postures")),
                (("nodes", "events"), ("builder", "events")),
                (("nodes", "balls"), ("builder", "balls")),
                (("builder", "frame_data"), ("data_writer", "data")),
                (("builder", "frame_data"), ("nodes", "data_output")),
            ],
        ),
        "drawing": Module(
            name="DrawingModule",
            components={
                "draw_players": DrawComponent(),
                "draw_projections": DrawProjections(),
                "build_frame": FrameBuilderComponent()
            },
            nodes={
                # Inputs
                "frame_data": Node(),
                "video_frame": Node(),
                "map_frame": Node(),

                # Outputs
                "final_frame": Node()
            },
            links=[
                (("nodes", "frame_data"), ("draw_players", "datas")),
                (("nodes", "frame_data"), ("draw_projections", "datas")),
                (("nodes", "video_frame"), ("draw_players", "frame")),
                (("nodes", "map_frame"), ("draw_projections", "frame")),
                (("draw_players", "output_frame"), ("build_frame", "frame1")),
                (("draw_projections", "output_frame"), ("build_frame", "frame2")),
                (("build_frame", "output_frame"), ("nodes", "final_frame")),
            ]
        ),
        "recorder": RecorderComponent(
            Path("db"), "player_detection", (1920, 540)
        ),
        "displayer": OpenCVDisplayerComponent(label="Players"),
    },
    links=[
        (("camera", "frame"), ("player_position_tracking", "frame")),
        (("camera", "frame"), ("ball_detection_module", "frame")),

        # Data register module
        (("ball_detection_module", "balls"), ("data_register", "balls")),
        (("player_position_tracking", "tracked_players"), ("data_register", "tracked_players")),
        (("player_position_tracking", "events"), ("data_register", "events")),

        # Drawing module
        (("camera", "frame"), ("drawing", "video_frame")),
        (("image", "image"), ("drawing", "map_frame")),
        (("data_register", "data_output"), ("drawing", "frame_data")),

        # Output
        (("drawing", "final_frame"), ("recorder", "frame")),
        (("drawing", "final_frame"), ("displayer", "frame")),
    ],
)

if __name__ == "__main__":
    export_plantuml(PlayerDetectionPipeline, Path("db"))
    PlayerDetectionPipeline()
