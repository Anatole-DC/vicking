from dataclasses import dataclass

from river import Data
from numpy import ndarray


@dataclass
class FrameData(Data):
    image: ndarray
