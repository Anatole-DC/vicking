from river import Pipeline

from vicking.components import OpenCVDisplayerComponent, USBCameraComponent

USBCameraPipeline = Pipeline(
    name="USBCameraPipeline",
    components={
        "camera": USBCameraComponent(),
        "displayer": OpenCVDisplayerComponent("USBCameraPipeline"),
    },
    links=[(("camera", "frame"), ("displayer", "frame"))],
)

if __name__ == "__main__":
    USBCameraPipeline()
