from dataclasses import dataclass, field
from typing import Any, Dict, List

from .data_manager import DataManagerComponent


@dataclass
class DataReaderComponent(DataManagerComponent):
    def __call__(self, args: List = ..., kwargs: Dict = ...) -> Any:
        raise NotImplementedError()
