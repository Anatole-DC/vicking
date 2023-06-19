from enum import Enum


class ProjectLectureMode(Enum):
    READ_ONLY: int = 0
    WRITE_ADD: int = 1
    WRITE_FROMSCRATCH: int = 2
