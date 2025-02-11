from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from sensor_fusion_exercise.utils import Degrees, Image, LocationNE, Pixels, SensorId

T = TypeVar("T")


class Sensor(ABC, Generic[T]):
    """A sensor device which perceives the environment."""

    @abstractmethod
    def next_tick(self) -> T:
        """
        Provides data `T` from the next tick which reflects a discrete time step,
        i.e. the next sensor output at time t+1.
        """
        ...

    @property
    @abstractmethod
    def location_north_east(self) -> LocationNE:
        """Sensor location in north-east inertial coordinates."""
        ...

    @property
    @abstractmethod
    def bearing(self) -> Degrees:
        """Heading/bearing in degrees with 0/360° facing north, 90° east,  180° south, 270° west."""
        ...

    @property
    @abstractmethod
    def sensor_id(self) -> int:
        """Unique sensor ID."""
        ...


@dataclass(frozen=True)
class CameraConfig:
    """
    Config for the camera parameters.

    Image coordinates are defined as:
    (0,0) being the top-left corner of the image and (image_width, image_height) the bottom-right corner.
    The field of view defines the angle of the cone of a camera's visible area.
    """

    image_width: Pixels
    image_height: Pixels
    fov_horizontal: Degrees
    fov_vertical: Degrees


@dataclass
class Frame:
    """The output of a camera at each time step."""

    image: Image
    fov_horizontal: Degrees
    fov_vertical: Degrees


class MockCamera(Sensor[Frame]):
    """
    A mocked camera which is attached to an asset and can pan around its z-axis with the current pan configuration
    stored in `bearing_angle`.
    """

    def __init__(
        self, config: CameraConfig, asset_location_ne: LocationNE, bearing_angle: Degrees, sensor_id: SensorId
    ) -> None:
        self._config = config
        self._asset_location = asset_location_ne
        self._bearing_angle = bearing_angle
        self._sensor_id = sensor_id

    def next_tick(self) -> Frame:
        image = np.zeros((self._config.image_height, self._config.image_width, 3), dtype=np.float32)
        return Frame(image=image, fov_horizontal=self._config.fov_horizontal, fov_vertical=self._config.fov_vertical)

    @property
    def location_north_east(self) -> LocationNE:
        """Camera/asset location in north-east inertial coordinates."""
        return self._asset_location

    @property
    def bearing(self) -> Degrees:
        """Camera heading/bearing in degrees with 0/360° facing north, 180° facing south and 90° facing east."""
        return self._bearing_angle

    @property
    def sensor_id(self) -> SensorId:
        return self._sensor_id
