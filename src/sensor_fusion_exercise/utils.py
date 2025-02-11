from __future__ import annotations

from dataclasses import dataclass

from nptyping import Float, NDArray, Shape

Degrees = float
Pixels = int
Meters = float
SensorId = int
Image = NDArray[Shape["H, W, 3"], Float]  # height x width x channel

OBJECT_PRIORS_WIDTH: dict[str, Meters] = {"tank": 7.0, "car": 3.5}


@dataclass
class LocationNE:
    """Location in north-east w.r.t. a flat surface, ie elevation (z-axis) is neglected."""

    north: Meters
    east: Meters

    @staticmethod
    def null() -> LocationNE:
        return LocationNE(north=0, east=0)
