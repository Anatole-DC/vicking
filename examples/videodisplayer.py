from pathlib import Path

from river import Pipeline

from vicking.components import OpenCVDisplayerComponent, VideoComponent


VideoDisplayerPipeline = Pipeline(
    name="VideoDisplayerPipeline",
    components={
        "camera": VideoComponent(
            Path(
                "/home/anatole/Projets/ocular/vicking_old/assets/videos/basketball.mp4"
            )
        ),
        "displayer": OpenCVDisplayerComponent("VideoDisplayerPipeline"),
    },
    links=[(("camera", "frame"), ("displayer", "frame"))],
)

if __name__ == "__main__":
    VideoDisplayerPipeline()
