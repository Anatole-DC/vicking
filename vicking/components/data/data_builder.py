from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from river import Component, Node, Data


@dataclass
class DataBuilderComponent(Component):
    output_model: Type

    def __post_init__(self):
        super().__post_init__()
        self.nodes["output"] = Node()

    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        output_data: Dict[str, Data] = {}
        for key, node in self.nodes.items():
            if key == "output":
                continue

            value: Data = node.get()
            if value is None:
                return
            print(type(value))
            output_data[key] = value

        print(output_data)

        self.nodes["output"].set(self.output_model(**output_data))

        print(self.nodes["output"])
