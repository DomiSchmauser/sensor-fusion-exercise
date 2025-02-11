from collections.abc import Callable

import matplotlib.pyplot as plt

from sensor_fusion_exercise.object_recognition import IdentifiedObject, MockObjectRecognition
from sensor_fusion_exercise.sensor import Frame, Sensor
from sensor_fusion_exercise.utils import SensorId


class FusionStation:
    """
    Central fusion station which gets frames from a sensor array at each tick, processes them using an
    object recognition algorithm and fuses localized objects from each camera into a single common representation
    visualised in a common operational picture.
    """

    def __init__(
        self,
        sensor_array: list[Sensor[Frame]],
        object_recognition: MockObjectRecognition,
        match_and_fuse_identified_objects: Callable[[dict[SensorId, list[IdentifiedObject]]], list[IdentifiedObject]],
    ) -> None:
        self._sensor_array = sensor_array
        self._object_recognition = object_recognition
        self._match_and_fuse_identified_objects = match_and_fuse_identified_objects

    def _next_tick(self) -> dict[SensorId, list[IdentifiedObject]]:
        """
        Advances a tick: gets frames from a sensor array and processes them using an
        object recognition algorithm. Returns a mapping of sensors to object identifications.
        """
        sensor_object_mapping = dict()
        for sensor in self._sensor_array:
            frame = sensor.next_tick()
            identified_objects = self._object_recognition.identify_and_localise(frame, sensor)
            sensor_object_mapping[sensor.sensor_id] = identified_objects
        return sensor_object_mapping

    def execute_without_fusion(self) -> dict[SensorId, list[IdentifiedObject]]:
        """Advances the `_next_tick` and visualises the not fused object identifications."""
        sensor_object_mapping = self._next_tick()
        not_fused_objects = [
            per_sensor_identified_object
            for per_sensor_identified_objects in sensor_object_mapping.values()
            for per_sensor_identified_object in per_sensor_identified_objects
        ]
        self.visualise_operational_picture(not_fused_objects)
        return sensor_object_mapping

    def execute_with_fusion(self) -> None:
        """Advances the `_next_tick`, fuses identifications and visualises the fused representation."""
        sensor_object_mapping = self._next_tick()
        fused_objects = self._match_and_fuse_identified_objects(sensor_object_mapping)
        self.visualise_operational_picture(fused_objects)

    def visualise_operational_picture(self, fused_objects: list[IdentifiedObject]) -> None:
        # Sensor locations
        north_sensors = [sensor.location_north_east.north for sensor in self._sensor_array]
        east_sensors = [sensor.location_north_east.east for sensor in self._sensor_array]
        marker_sensors = ["g"] * len(self._sensor_array)
        annotation_sensors = [f"Camera {sensor.sensor_id}" for sensor in self._sensor_array]

        # Localised objects
        north_objects = [fused.object_location_north_east.north for fused in fused_objects]
        east_objects = [fused.object_location_north_east.east for fused in fused_objects]
        marker_objects = ["r"] * len(fused_objects)
        annotation_objects = [fused.class_name for fused in fused_objects]

        # Combined
        combined_east = east_sensors + east_objects
        combined_north = north_sensors + north_objects

        fig, ax = plt.subplots()
        ax.scatter(combined_east, combined_north, s=100.0, c=marker_sensors + marker_objects)
        for i, txt in enumerate(annotation_sensors + annotation_objects):
            ax.annotate(txt, (combined_east[i], combined_north[i]))
        ax.set_ylabel("North [meters]")
        ax.set_xlabel("East [meters]")
        ax.set_xlim(-5.0, 80.0)
        ax.set_ylim(-5.0, 160.0)
        ax.grid()

        plt.show()
