from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sensor_fusion_exercise.sensor import Frame, Sensor
from sensor_fusion_exercise.utils import LocationNE, Meters, Pixels, SensorId


@dataclass
class BoundingBox:
    x: Pixels
    y: Pixels
    width: Pixels
    height: Pixels


@dataclass
class IdentifiedObject:
    bounding_box: BoundingBox
    class_name: str  # object class
    object_location_north_east: LocationNE


class MockObjectRecognition:
    """
    Mocked object recognition algorithm using provided ground truth detections with classes and bounding boxes.
    A projected north-east object location is derived from a monocular distance estimation and the bearing direction
    of the object.
    """

    def __init__(
        self,
        ground_truth_objects: dict[SensorId, list[IdentifiedObject]],
        monocular_distance_estimation: Callable[[IdentifiedObject, Frame], Meters],
    ):
        self._gt_objects = ground_truth_objects
        self._monocular_distance_estimation = monocular_distance_estimation

    @staticmethod
    def bearing_and_distance_to_north_east(
        sensor: Sensor[Frame], frame: Frame, identified_object: IdentifiedObject, object_distance_wrt_sensor: Meters
    ) -> LocationNE:
        """
        Calculates the north-east location of an object by chaining the sensor location and bearing direction
        with the identified object with its distance estimate.
        """
        center_pixel_x_bbox = identified_object.bounding_box.x + (identified_object.bounding_box.width / 2)
        # Convert to relative coordinates [0-1]
        relative_center_x_bbox = center_pixel_x_bbox / frame.image.shape[1]
        delta_from_sensor_bearing = relative_center_x_bbox * frame.fov_horizontal - frame.fov_horizontal / 2

        # Convert polar coordinates to cartesian coordinates
        combined_bearing = sensor.bearing + delta_from_sensor_bearing
        north_wrt_sensor = object_distance_wrt_sensor * np.cos(np.deg2rad(combined_bearing))
        east_wrt_sensor = object_distance_wrt_sensor * np.sin(np.deg2rad(combined_bearing))
        # Add sensor location to get the location in the global coordinate system
        return LocationNE(
            north=north_wrt_sensor + sensor.location_north_east.north,
            east=east_wrt_sensor + sensor.location_north_east.east,
        )

    def identify_and_localise(self, frame: Frame, sensor: Sensor[Frame]) -> list[IdentifiedObject]:
        """Identifies and localises an object from a given frame."""
        identified_objects: list[IdentifiedObject] = []
        for identified_object in self._gt_objects.get(sensor.sensor_id):  # type: ignore[union-attr]
            object_distance_wrt_sensor = self._monocular_distance_estimation(identified_object, frame)
            identified_object.object_location_north_east = self.bearing_and_distance_to_north_east(
                sensor, frame, identified_object, object_distance_wrt_sensor
            )
            identified_objects.append(identified_object)
        return identified_objects
