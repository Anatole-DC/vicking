from dataclasses import dataclass, make_dataclass, field

from cvermeer.shapes import Point

from .projection_map import ProjectionMapComponent
from vicking.components.detection.openpifpaf_detector import PosturesData

@dataclass
class PostureProjectionMapComponent(ProjectionMapComponent):

    def __call__(self):
        postures_data = self.nodes["points"].get()

        if postures_data is None: return

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
        self.nodes["projections"].set(postures_data)