from dataclasses import dataclass

from .capture import CaptureComponent


@dataclass
class USBCameraComponent(CaptureComponent):
    channel: int = 0

    def __post_init__(self):
        try:
            self.channel = int(self.channel)
        except ValueError as error:
            raise ValueError(
                f"Channel must be an integer. Argument of type {type(self.channel)} received."
            )
        super().__post_init__()
