"""
Device utilities - Re-exports from homebench_devices for backward compatibility.

All device implementations are in homebench_devices.py which provides
the full HomeBench device classes.
"""

# Re-export all device classes from homebench_devices
from src.utils.homebench_devices import (
    # Device classes
    LightDevice,
    AirConditionerDevice,
    HeatingDevice,
    FanDevice,
    GarageDoorDevice,
    BlindsDevice,
    CurtainDevice,
    AirPurifiersDevice,
    WaterHeaterDevice,
    MediaPlayerDevice,
    VacuumRobotrDevice,
    AromatherapyDevice,
    TrashDevice,
    HumidifierDevice,
    DehumidifiersDevice,
    PetFeederDevice,
    BedDevice,
    # Device map
    device_map,
    # Room list
    AllCandiateRoom,
)

__all__ = [
    "LightDevice",
    "AirConditionerDevice",
    "HeatingDevice",
    "FanDevice",
    "GarageDoorDevice",
    "BlindsDevice",
    "CurtainDevice",
    "AirPurifiersDevice",
    "WaterHeaterDevice",
    "MediaPlayerDevice",
    "VacuumRobotrDevice",
    "AromatherapyDevice",
    "TrashDevice",
    "HumidifierDevice",
    "DehumidifiersDevice",
    "PetFeederDevice",
    "BedDevice",
    "device_map",
    "AllCandiateRoom",
]
