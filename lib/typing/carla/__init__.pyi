"""Minimal stub for the carla package, covering only the APIs this repo uses.

The carla 0.9.15 wheel re-exports a binary module without stubs, so every
attribute is unknown to static analysis without this file.
"""

class Location:
    x: float
    y: float
    z: float

class Rotation:
    pitch: float
    yaw: float
    roll: float

class Transform:
    location: Location
    rotation: Rotation

class VehicleControl:
    throttle: float
    steer: float
    brake: float
    hand_brake: bool
    reverse: bool
    manual_gear_shift: bool
    gear: int
    def __init__(
        self,
        throttle: float = ...,
        steer: float = ...,
        brake: float = ...,
        hand_brake: bool = ...,
        reverse: bool = ...,
        manual_gear_shift: bool = ...,
        gear: int = ...,
    ) -> None: ...
