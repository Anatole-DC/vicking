from dataclasses import dataclass, asdict
from pathlib import Path
import json

from tinydb import TinyDB
from tinydb.table import Table, Document

from .project_lecture_mode import ProjectLectureMode


@dataclass
class Project:
    name: str
    path: Path
    video_path: Path
    lecture_mode: ProjectLectureMode = ProjectLectureMode.WRITE_FROMSCRATCH

    def __post_init__(self):
        # Check video validity
        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Path to video '{self.video_path.absolute()}' does not exist"
            )
        if not self.video_path.is_file():
            raise TypeError(
                f"Video path '{self.video_path.absolute()}' is not a file path"
            )

        # Retrieve or create output
        if not self.path.exists():
            print(
                f"Path to project output {self.path.absolute()} does not exist, therefore will be created"
            )
            self.path.mkdir()
        elif self.path.is_file():
            raise TypeError(
                f"Project output path '{self.path.absolute()}' must be a folder"
            )

        if not self.data_path.exists():
            if self.lecture_mode is ProjectLectureMode.READ_ONLY:
                raise Exception(
                    "Project is read-only but the the data path given does not exist."
                )
            open(self.data_path, "w").write(
                json.dumps(
                    {
                        "name": self.name,
                        "video": str(self.video_path.absolute()),
                        "output": str(self.data_path.absolute()),
                    }
                )
            )

        self.database: TinyDB = TinyDB(self.data_path)

        self.frame_data: Table = self.database.table("frame_data")
        self.event_data: Table = self.database.table("event_data")

    @property
    def data_path(self) -> Path:
        return self.path / f"{self.name}.json"

    def write(self, frame_id: int, data) -> None:
        self.frame_data.upsert(Document(asdict(data), frame_id))

    def read(self, frame_id: int) -> Document:
        return self.frame_data.get(doc_id=frame_id)
