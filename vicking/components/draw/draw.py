from dataclasses import dataclass, field
from typing import Any, Dict, List

from river import Component, Node, Data
from cvermeer.drawing import Drawable
from numpy import ndarray

from vicking.data.frame_data import FrameData


@dataclass
class DrawComponent(Component):
    def __post_init__(self):
        super().__post_init__()
        self.nodes = {"frame": Node(), "datas": Node(), "output_frame": Node()}

    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        frame: FrameData = self.nodes["frame"].get()
        datas: Data = self.nodes["datas"].get()

        if frame is None or datas is None:
            return

        self.draw_data(datas, frame.image)

        self.nodes["output_frame"].set(frame)

    def draw_data(self, content, frame: ndarray) -> None:
        if isinstance(content, List):
            return [self.draw_data(data, frame) for data in content]
        if not isinstance(content, Drawable):
            raise TypeError(
                f"{self.name}'s datas expects Drawables. Not {type(content)}"
            )
        content.draw(frame)
