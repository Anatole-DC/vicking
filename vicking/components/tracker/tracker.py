from dataclasses import dataclass, field, make_dataclass
from typing import Any, Dict, List, Type
from copy import deepcopy

from river import Component, Node, Data
from vicking.models import Trajectory
from cvermeer.colors import Color


@dataclass
class TrackedObjectsData(Data):
    objects: List

    @property
    def data(self):
        return self.objects

@dataclass
class TrajectoriesData(Data):
    tajectories: List[Trajectory]

    @property
    def data(self):
        return self.tajectories


@dataclass
class TrackerComponent(Component):
    model: Type
    trajectory_buffer: List[Trajectory] = field(init=False, default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.nodes = {
            "detections": Node(),
            "trajectories": Node(),
            "tracked_objects": Node(),
        }

    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        detections: Data = self.nodes["detections"].get()
        if detections is None: return

        self.trajectory_buffer = self.track(detections.data)

        tracked_objects: List = []
        for trajectory in self.trajectory_buffer:
            tracked_object = deepcopy(trajectory.last())
            tracked_object.__class__ = make_dataclass(
                        f"Tracked{tracked_object.__class__.__name__}",
                        fields=[
                            ("uuid", int, field(default=trajectory.uuid)),
                            ("color", Color, field(default=trajectory.color))
                        ],
                        bases=(self.model,),
                    )
            tracked_objects.append(tracked_object)

        self.nodes["tracked_objects"].set(
            TrackedObjectsData(
                detections.token,
                tracked_objects
            )
        )

        for trajectory in self.trajectory_buffer:
            if trajectory.is_lost():
                self.trajectory_buffer.remove(trajectory)

    def track(self, detections) -> List[Trajectory]:
        ...
