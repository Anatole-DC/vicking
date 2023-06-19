from dataclasses import dataclass, field
from typing import List, Tuple
import logging
from copy import deepcopy

import cv2 as opencv
import numpy
from river import Component, Node
from cvermeer.shapes import Point
from cvermeer.colors import Color


POINTS_COLOR: Tuple[Color] = (
    Color(0, 255, 0),
    Color(0, 0, 255),
    Color(255, 0, 0),
    Color(150, 150, 150)
)


@dataclass
class ProjectionMapComponent(Component):
    from_image: numpy.ndarray
    to_map: numpy.ndarray
    points_on_image: List[Point] = field(default_factory=list)
    points_on_map: List[Point] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.nodes["points"] = Node()
        self.nodes["projections"] = Node()

        print(self.points_on_map)

        # if len(self.points_on_image) != 4 or len(self.points_on_map) != 4:
        #     self.calibrate()

        self.points_on_image = [point.to_scale(self.from_image) for point in self.points_on_image]
        self.points_on_map = [point.to_scale(self.to_map) for point in self.points_on_map]

        self.transformation_matrix = opencv.getPerspectiveTransform(
            numpy.float32([point.tuple() for point in self.points_on_image]),
            numpy.float32([point.tuple() for point in self.points_on_map])
        )

    def __call__(self):
        points: List[Point] = self.nodes["points"].get()

        if points is None: return

        projected_points: List[Point] = []

        for point in points:
            if point is None: continue
            projected_points.append(self.project_point(point))

        self.nodes["points"].set(projected_points)

    def project_point(self, point: Point) -> Point:
        point_arr = numpy.array([[[point.x, point.y]]], dtype=numpy.float32)
        transformed_point_arr = opencv.perspectiveTransform(point_arr, self.transformation_matrix)
        transformed_point = Point(float(transformed_point_arr[0][0][0]), float(transformed_point_arr[0][0][1]))
        return transformed_point
    
    def is_point_in_field(self, point: Point) -> bool:
        min_x: int = min([point.x for point in self.points_on_map])
        min_y: int = min([point.y for point in self.points_on_map])
        max_x: int = max([point.x for point in self.points_on_map])
        max_y: int = max([point.y for point in self.points_on_map])

        return (min_x <= point.x <= max_x) and (min_y <= point.y <= max_y)

    def calibrate(self):
        opencv.namedWindow("ProjectionImage", opencv.WINDOW_NORMAL)
        opencv.namedWindow("ProjectionMap", opencv.WINDOW_NORMAL)

        opencv.imshow("ProjectionImage", self.from_image.image)
        opencv.imshow("ProjectionMap", self.to_map.image)


        opencv.setMouseCallback('ProjectionImage', self.call_back_image)
        opencv.setMouseCallback('ProjectionMap', self.call_back_map)

        opencv.waitKey(0)
        opencv.destroyWindow("ProjectionImage")
        opencv.destroyWindow("ProjectionMap")

        print(f"Points are :\n  * Image : {self.points_on_image}\n  * Map   : {self.points_on_map}")


    def call_back_image(self, event, x, y, flags, params):
        frame = deepcopy(self.from_image)

        if event == opencv.EVENT_LBUTTONDOWN:
            if len(self.points_on_image) == 4:
                logging.error("All points are given for image !")
            else:
                self.points_on_image.append(Point(x, y).from_scale(frame))

        if event == opencv.EVENT_MBUTTONDOWN:
            if not self.points_on_image:
                logging.error("Can't remove points from empty image point list !")
            else:
                self.points_on_image.pop(-1)

        [point.draw(frame, POINTS_COLOR[index], 10)
         for index, point in enumerate(self.points_on_image)]

        opencv.imshow("ProjectionImage", frame.image)

    def call_back_map(self, event, x, y, flags, params):
        frame = deepcopy(self.to_map)

        if event == opencv.EVENT_LBUTTONDOWN:
            if len(self.points_on_image) == 4:
                logging.error("All points are given for map !")
            else:
                self.points_on_map.append(Point(x, y).from_scale(frame))

        if event == opencv.EVENT_MBUTTONDOWN:
            if not self.points_on_map:
                logging.error("Can't remove points from empty map point list !")
            else:
                self.points_on_map.pop(-1)

        [point.draw(frame, POINTS_COLOR[index], 10)
         for index, point in enumerate(self.points_on_map)]

        opencv.imshow("ProjectionMap", frame.image)
