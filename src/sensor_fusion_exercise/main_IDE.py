import logging
import math

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from sensor_fusion_exercise.fusion_station import FusionStation
from sensor_fusion_exercise.object_recognition import BoundingBox, IdentifiedObject, MockObjectRecognition
from sensor_fusion_exercise.sensor import CameraConfig, Frame, MockCamera, Sensor
from sensor_fusion_exercise.utils import OBJECT_PRIORS_WIDTH, LocationNE, Meters, SensorId

logger = logging.getLogger(__name__)


def monocular_distance_from_object_prior(identified_object: IdentifiedObject, frame: Frame) -> Meters:
    """
    Calculates the distance based on an object size prior and the respective bounding box size.
    Returns a distance estimate from the camera to the target object in meters.
    """
    image_width = frame.image.shape[1]  # Retrieve width pixels dimension from image in the format H x W x C
    bbox_width_degree = identified_object.bounding_box.width / image_width * frame.fov_horizontal
    distance = Meters(
        OBJECT_PRIORS_WIDTH[identified_object.class_name] / (2.0 * np.tan(np.deg2rad(bbox_width_degree) / 2.0))
    )
    return distance


def match_and_fuse_identified_objects(
    sensor_object_mapping: dict[SensorId, list[IdentifiedObject]],
) -> list[IdentifiedObject]:
    """
    Matches identified objects of first sensor against the identified objects of the second sensor.
    First, a distance metric is defined to derive the similarity/dissimilarity of object pairs.
    Based on the distance metrics, one can find an optimal assignment between object pairs to
    output a list of fused identified objects.
    """

    def distance_metric(object1: IdentifiedObject, object2: IdentifiedObject) -> Meters:
        # Euclidian distance
        euclidian_distance = math.sqrt(
            (object2.object_location_north_east.north - object1.object_location_north_east.north) ** 2
            + (object2.object_location_north_east.east - object1.object_location_north_east.east) ** 2
        )
        return Meters(euclidian_distance)

    # Generate cost matrix
    cost_matrix = np.zeros((2, 2))
    sensor_1_objects = sensor_object_mapping[1]
    sensor_2_objects = sensor_object_mapping[2]
    # rows = sensor 1, columns = sensor2
    for row_idx, o1 in enumerate(sensor_1_objects):
        for col_idx, o2 in enumerate(sensor_2_objects):
            cost_matrix[row_idx, col_idx] = distance_metric(o1, o2)

    # Bipartite graph matching
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    fused_objects = []
    for row, col in zip(row_idx, col_idx, strict=True):  # type: ignore[call-overload]
        s1_object = sensor_1_objects[row]
        s2_object = sensor_2_objects[col]
        logger.info(
            f"Fused sensor 1 identified object at location {s1_object.object_location_north_east} and "
            f"sensor 2 identified object at location {s2_object.object_location_north_east}"
        )

        # Re-calculate the fused location as the average from both perspectives
        fused_north = (s1_object.object_location_north_east.north + s2_object.object_location_north_east.north) / 2.0
        fused_east = (s1_object.object_location_north_east.east + s2_object.object_location_north_east.east) / 2.0
        fused_location = LocationNE(north=fused_north, east=fused_east)
        s1_object.object_location_north_east = fused_location
        fused_objects.append(s1_object)

    return fused_objects


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera_cfg = CameraConfig(image_width=1920, image_height=1080, fov_horizontal=40.0, fov_vertical=22.5)

    sensor_1 = MockCamera(
        config=camera_cfg, asset_location_ne=LocationNE(north=0.0, east=0.0), bearing_angle=26.6, sensor_id=1
    )
    sensor_2 = MockCamera(
        config=camera_cfg, asset_location_ne=LocationNE(north=0.0, east=50.0), bearing_angle=0.0, sensor_id=2
    )

    sensor_array: list[Sensor[Frame]] = [sensor_1, sensor_2]

    ground_truth_objects = {
        1: [
            IdentifiedObject(
                class_name="tank",
                bounding_box=BoundingBox(880, 500, 180, 100),
                object_location_north_east=LocationNE.null(),
            ),
            IdentifiedObject(
                class_name="car",
                bounding_box=BoundingBox(800, 500, 60, 100),
                object_location_north_east=LocationNE.null(),
            ),
        ],
        2: [
            IdentifiedObject(
                class_name="car",
                bounding_box=BoundingBox(1200, 500, 70, 100),
                object_location_north_east=LocationNE.null(),
            ),
            IdentifiedObject(
                class_name="tank",
                bounding_box=BoundingBox(860, 500, 180, 100),
                object_location_north_east=LocationNE.null(),
            ),
        ],
    }

    object_recognition = MockObjectRecognition(
        ground_truth_objects=ground_truth_objects, monocular_distance_estimation=monocular_distance_from_object_prior
    )

    # First, lets visualise the common operational picture if we don't fuse identified objects and print the sensor object mapping from the object recognition
    fusion_station = FusionStation(sensor_array, object_recognition, match_and_fuse_identified_objects)
    sensor_object_mapping = fusion_station.execute_without_fusion()

    logger.info("Sensor object mapping: \n")
    for items in sensor_object_mapping.items():
        logger.info(f"Sensor ID: {items[0]} with identified objects: {items[1]} \n")

    fusion_station.execute_with_fusion()
