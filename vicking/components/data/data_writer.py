from dataclasses import dataclass, field
from typing import List, Dict, Any

from river import Data

from .data_manager import DataManagerComponent


@dataclass
class DataWriterComponent(DataManagerComponent):
    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        datas: Data = self.nodes["data"].get()

        if datas is None:
            return

        self.project.write(datas.token, datas)
