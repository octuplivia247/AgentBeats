"""
Smart Home Environment Manager

This module provides the SmartHomeEnvironment class that wraps SmartHomeEnvManager
from manager2.py to provide a unified interface for the evaluation framework.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.green_agent.core.manager2 import (
    DeviceSpec,
    HomeConfig,
    SmartHomeEnvManager,
    load_home_from_jsonl,
)


@dataclass
class ActionRecord:
    """Record of a device action execution."""

    timestamp: str
    device_name: str  # Format: "room_name.device_name"
    action: str
    parameters: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    result: Any = None


class SmartHomeEnvironment:
    """
    Manages smart home devices and their states during evaluation.

    This class wraps SmartHomeEnvManager to provide:
    - Device initialization from config or JSONL files
    - Action execution with logging
    - State tracking for evaluation

    Usage:
        # From config dict
        env = SmartHomeEnvironment()
        env.initialize(config)

        # From JSONL file
        env = SmartHomeEnvironment.from_jsonl("data/home_status_method.jsonl")

        # Execute actions
        result = env.execute_action("living_room.light", "turn_on")
        state = env.get_device_state("living_room.light")
    """

    def __init__(self, home_id: int = 1):
        """
        Initialize the environment manager.

        Args:
            home_id: Unique identifier for this home environment
        """
        self.home_id = home_id
        self._manager: Optional[SmartHomeEnvManager] = None
        self.devices: Dict[str, Any] = {}
        self.device_states: Dict[str, Dict[str, Any]] = {}
        self.action_log: List[ActionRecord] = []
        self.initialized = False
        self._initial_config: Optional[Dict[str, Any]] = None

    def _normalize_device_name(self, device_name: str) -> str:
        """
        Normalize device name to room.device format.
        
        Handles:
        - living_room_light -> living_room.light
        - living_room.light -> living_room.light (unchanged)
        - master_bedroom_air_conditioner -> master_bedroom.air_conditioner
        """
        if "." in device_name:
            return device_name
        
        # Known device types to look for at the end
        device_types = [
            "light", "air_conditioner", "heating", "fan", "garage_door",
            "blinds", "curtain", "air_purifiers", "water_heater", 
            "media_player", "vacuum_robot", "aromatherapy", "trash",
            "humidifier", "dehumidifiers", "pet_feeder", "thermostat"
        ]
        
        # Try to find a device type suffix
        for dt in device_types:
            suffix = f"_{dt}"
            if device_name.endswith(suffix):
                room_part = device_name[:-len(suffix)]
                return f"{room_part}.{dt}"
        
        # Fallback: replace last underscore with dot
        if "_" in device_name:
            last_underscore = device_name.rfind("_")
            return device_name[:last_underscore] + "." + device_name[last_underscore + 1:]
        
        return device_name

    def _find_device(self, device_name: str) -> Optional[Any]:
        """
        Find a device by name, trying multiple formats.
        """
        # Try exact match first
        if device_name in self.devices:
            return self.devices[device_name]
        
        # Try normalized name
        normalized = self._normalize_device_name(device_name)
        if normalized in self.devices:
            return self.devices[normalized]
        
        # Try with underscores replaced by dots
        with_dots = device_name.replace("_", ".")
        if with_dots in self.devices:
            return self.devices[with_dots]
        
        # Try partial match (room.device where device matches)
        for key in self.devices:
            if "." in key:
                _, dev = key.split(".", 1)
                if dev == device_name or device_name.endswith(dev):
                    return self.devices[key]
        
        return None

    @classmethod
    def from_jsonl(
        cls, method_path: str, status_path: Optional[str] = None
    ) -> "SmartHomeEnvironment":
        """
        Create environment from JSONL files.

        Args:
            method_path: Path to home_status_method.jsonl
            status_path: Optional path to home_status_data.jsonl (uses same file if not provided)

        Returns:
            Initialized SmartHomeEnvironment
        """
        env = cls()
        env._manager = load_home_from_jsonl(method_path)
        env.home_id = env._manager.home_id
        env.initialized = True

        # Build device references
        for room_name in env._manager.list_rooms():
            for device_name in env._manager.list_devices(room_name):
                full_name = f"{room_name}.{device_name}"
                device = env._manager.get_device(room_name, device_name)
                env.devices[full_name] = device
                if hasattr(device, "get_status"):
                    env.device_states[full_name] = device.get_status()

        return env

    def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the smart home environment with devices from configuration.

        Supports two config formats:
        1. Simple format (for testing):
           {
               "rooms": ["living_room", "bedroom"],
               "devices": {
                   "living_room_light": {"type": "light", "state": "off"},
                   ...
               }
           }

        2. HomeBench format (from JSONL):
           {
               "home_id": 0,
               "method": [...],
               "home_status": {...}
           }

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary with initialization status
        """
        self._initial_config = config
        self.action_log.clear()

        # Check if HomeBench format
        if "home_status" in config and "method" in config:
            return self._initialize_from_homebench(config)
        else:
            return self._initialize_simple(config)

    def _initialize_from_homebench(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize from HomeBench JSONL format."""
        methods = [
            DeviceSpec(
                room_name=m["room_name"],
                device_name=m["device_name"],
                operation=m["operation"],
                parameters=m.get("parameters", []),
            )
            for m in config["method"]
        ]

        home_config = HomeConfig(
            home_id=config.get("home_id", self.home_id),
            methods=methods,
            home_status=config["home_status"],
        )

        self._manager = SmartHomeEnvManager(home_config)
        self.home_id = self._manager.home_id
        self.initialized = True

        # Build device references
        self.devices.clear()
        self.device_states.clear()

        for room_name in self._manager.list_rooms():
            for device_name in self._manager.list_devices(room_name):
                full_name = f"{room_name}.{device_name}"
                device = self._manager.get_device(room_name, device_name)
                self.devices[full_name] = device
                if hasattr(device, "get_status"):
                    self.device_states[full_name] = device.get_status()

        return {
            "status": "initialized",
            "home_id": self.home_id,
            "device_count": len(self.devices),
            "devices": list(self.devices.keys()),
            "rooms": self._manager.list_rooms(),
        }

    def _initialize_simple(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize from simple config format (for testing/demos)."""
        from src.utils.homebench_devices import device_map

        self.devices.clear()
        self.device_states.clear()

        devices_config = config.get("devices", {})

        for device_key, device_info in devices_config.items():
            device_type = device_info.get("type", "light")
            initial_state = device_info.get("state", "off")

            # Get device class from map
            device_cls = device_map.get(device_type)
            if device_cls is None:
                continue

            # Create device instance
            try:
                device = device_cls(state=initial_state)
            except TypeError:
                try:
                    device = device_cls()
                except Exception:
                    continue

            # Apply attributes
            if "attributes" in device_info and hasattr(device, "attributes"):
                for attr_name, attr_value in device_info["attributes"].items():
                    if attr_name in device.attributes:
                        if isinstance(attr_value, dict) and "value" in attr_value:
                            device.attributes[attr_name]["value"] = attr_value["value"]
                        else:
                            device.attributes[attr_name]["value"] = attr_value

            self.devices[device_key] = device
            if hasattr(device, "get_status"):
                self.device_states[device_key] = device.get_status()

        self.initialized = True

        return {
            "status": "initialized",
            "home_id": self.home_id,
            "device_count": len(self.devices),
            "devices": list(self.devices.keys()),
        }

    def execute_action(
        self, device_name: str, action: str, parameters: Optional[Dict[str, Any]] = None
    ) -> ActionRecord:
        """
        Execute an action on a device and log the result.

        Args:
            device_name: Device name (e.g., "living_room.light" or "living_room_light")
            action: Action to perform (e.g., "turn_on", "set_brightness")
            parameters: Optional parameters for the action

        Returns:
            ActionRecord with execution details
        """
        parameters = parameters or {}
        timestamp = datetime.now().isoformat()

        # Normalize device name - try multiple formats
        normalized_name = self._normalize_device_name(device_name)

        # Try to find device with various name formats
        device = self._find_device(device_name)

        if device is None:
            # Build helpful error message
            available = list(self.devices.keys())[:10]  # First 10 devices
            error_msg = f"Device '{device_name}' not found. Tried: '{normalized_name}'. Available: {available}"
            record = ActionRecord(
                timestamp=timestamp,
                device_name=device_name,
                action=action,
                parameters=parameters,
                success=False,
                error=error_msg,
            )
            self.action_log.append(record)
            return record

        # Try to execute via manager if available
        if self._manager and "." in normalized_name:
            parts = normalized_name.split(".", 1)
            room_name, dev_name = parts[0], parts[1]
            try:
                result = self._manager.call(room_name, dev_name, action, **parameters)
                record = ActionRecord(
                    timestamp=timestamp,
                    device_name=normalized_name,
                    action=action,
                    parameters=parameters,
                    success=True,
                    result=result,
                )
                # Update state
                if hasattr(device, "get_status"):
                    self.device_states[normalized_name] = device.get_status()
                self.action_log.append(record)
                return record
            except Exception as e:
                record = ActionRecord(
                    timestamp=timestamp,
                    device_name=normalized_name,
                    action=action,
                    parameters=parameters,
                    success=False,
                    error=str(e),
                )
                self.action_log.append(record)
                return record

        # Fallback: direct method call on device
        if not hasattr(device, action):
            # Check operations dict
            if hasattr(device, "operations") and action in device.operations:
                try:
                    result = device.operations[action](**parameters)
                    record = ActionRecord(
                        timestamp=timestamp,
                        device_name=device_name,
                        action=action,
                        parameters=parameters,
                        success=True,
                        result=result,
                    )
                    if hasattr(device, "get_status"):
                        self.device_states[device_name] = device.get_status()
                    self.action_log.append(record)
                    return record
                except Exception as e:
                    record = ActionRecord(
                        timestamp=timestamp,
                        device_name=device_name,
                        action=action,
                        parameters=parameters,
                        success=False,
                        error=str(e),
                    )
                    self.action_log.append(record)
                    return record

            record = ActionRecord(
                timestamp=timestamp,
                device_name=device_name,
                action=action,
                parameters=parameters,
                success=False,
                error=f"Device '{device_name}' has no method or operation '{action}'",
            )
            self.action_log.append(record)
            return record

        # Direct method call
        try:
            method = getattr(device, action)
            result = method(**parameters)
            record = ActionRecord(
                timestamp=timestamp,
                device_name=device_name,
                action=action,
                parameters=parameters,
                success=True,
                result=result,
            )
            if hasattr(device, "get_status"):
                self.device_states[device_name] = device.get_status()
            self.action_log.append(record)
            return record
        except Exception as e:
            record = ActionRecord(
                timestamp=timestamp,
                device_name=device_name,
                action=action,
                parameters=parameters,
                success=False,
                error=str(e),
            )
            self.action_log.append(record)
            return record

    def get_device_state(self, device_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a specific device.

        Args:
            device_name: Device name (e.g., "living_room.light")

        Returns:
            Device state dictionary or None if device not found
        """
        # Try cached state first
        if device_name in self.device_states:
            return self.device_states[device_name]

        # Try to get fresh state
        device = self.devices.get(device_name)
        if device and hasattr(device, "get_status"):
            state = device.get_status()
            self.device_states[device_name] = state
            return state

        return None

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states of all devices in the environment.

        Returns:
            Dictionary mapping device names to their states
        """
        # Refresh all states
        for device_name, device in self.devices.items():
            if hasattr(device, "get_status"):
                self.device_states[device_name] = device.get_status()

        return dict(self.device_states)

    def get_action_log(self) -> List[ActionRecord]:
        """
        Get the log of all actions executed in this environment.

        Returns:
            List of ActionRecord objects in chronological order
        """
        return list(self.action_log)

    def reset(self) -> None:
        """
        Reset the environment to initial state.

        Clears action log and reinitializes from original config.
        """
        self.action_log.clear()

        if self._initial_config:
            self.initialize(self._initial_config)

    def get_tools_info(self) -> List[Dict[str, Any]]:
        """
        Get tool/device information for agent prompts.

        Returns:
            List of tool descriptions for each device
        """
        tools = []
        for device_name, device in self.devices.items():
            operations = []
            if hasattr(device, "operations"):
                operations = list(device.operations.keys())
            elif hasattr(device, "__class__"):
                # Get public methods
                operations = [
                    m for m in dir(device)
                    if not m.startswith("_") and callable(getattr(device, m))
                    and m not in ("get_status", "initialize", "random_initialize",
                                  "generate_instructions", "generate_unexist_instructions")
                ]

            tools.append({
                "device": device_name,
                "type": getattr(device, "name", device_name.split(".")[-1] if "." in device_name else "unknown"),
                "operations": operations,
                "state": self.device_states.get(device_name, {}),
            })

        return tools
