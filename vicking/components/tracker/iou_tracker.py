from dataclasses import dataclass
from typing import List

from cvermeer.shapes.helper import get_iou

from .tracker import TrackerComponent
from vicking.models import Trajectory


@dataclass
class IOUTrackerComponent(TrackerComponent):
    min_iou_confidence_threshold: float

    def track(self, detections) -> List[Trajectory]:
        if not self.trajectory_buffer:
            return [Trajectory([detection]) for detection in detections]

        final_trajectories: List[Trajectory] = []

        unmatched_detection_indexes: List[int] = [
            index for index, _ in enumerate(detections)
        ]
        detection_scores: List[List[float]] = []

        # Compute score for each detection
        for trajectory in self.trajectory_buffer:
            detection_scores.append(
                [get_iou(detection.bbox, trajectory.last().bbox) for detection in detections]
            )

        # Hungarian algorithm
        for traj_index, trajectory in enumerate(self.trajectory_buffer):
            max_score: float = 0.0
            max_index: int = None
            for det_index, score in enumerate(detection_scores[traj_index]):
                if score > max_score:
                    max_score = score
                    max_index = det_index
            if (max_score >= self.min_iou_confidence_threshold) and (
                max_index in unmatched_detection_indexes
            ):
                trajectory.update(detections[max_index])
                unmatched_detection_indexes.remove(max_index)
            final_trajectories.append(trajectory)

        # Finally, we create a trajectory for each unmatched detection
        for index in unmatched_detection_indexes:
            final_trajectories.append(Trajectory([detections[index]]))

        # Return the final trajectories
        return final_trajectories
