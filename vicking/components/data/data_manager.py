from dataclasses import dataclass, field

from river import Component, Node

from vicking.projects import Project


@dataclass
class DataManagerComponent(Component):
    project: Project

    def __post_init__(self):
        super().__post_init__()
        self.nodes = {"data": Node()}
