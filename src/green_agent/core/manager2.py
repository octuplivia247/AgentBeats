# smart_home_env_manager.py

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union, Optional

# Import all device classes from HomeBench
# Make sure HomeBench is on PYTHONPATH, e.g.:
#   export PYTHONPATH=/path/to/HomeBench:$PYTHONPATH
from src.utils.homebench_devices import *

# ---------- Data structures ----------

@dataclass
class DeviceSpec:
    room_name: str
    device_name: str
    operation: str
    parameters: List[Dict[str, str]]  # {"name": ..., "type": ...}


@dataclass
class HomeConfig:
    home_id: int
    methods: List[DeviceSpec]
    home_status: Dict[str, Any]


# ---------- Environment Manager ----------

class SmartHomeEnvManager:
    """
    Smart home environment manager that:
    - Builds device objects from HomeBench visual_home/device.py
    - Initializes them from a JSONL home_status snapshot
    - Executes operations (turn_on, set_temperature, etc.) with type-checked args
    """

    def __init__(self, config: HomeConfig):
        self.home_id = config.home_id
        self._methods_index: Dict[
            Tuple[str, str, str], DeviceSpec
        ] = {}  # (room, device, op) -> spec

        for spec in config.methods:
            key = (spec.room_name, spec.device_name, spec.operation)
            self._methods_index[key] = spec

        # room_name -> device_name -> device_instance
        self.rooms: Dict[str, Dict[str, Any]] = {}

        # Create and initialize devices from home_status
        self._build_devices(config.home_status)

    # ---------- Public API ----------

    def list_rooms(self) -> List[str]:
        return sorted(self.rooms.keys())

    def list_devices(self, room_name: str) -> List[str]:
        return sorted(self.rooms.get(room_name, {}).keys())

    def get_device(self, room_name: str, device_name: str) -> Any:
        return self.rooms[room_name][device_name]

    def call(
        self,
        room_name: str,
        device_name: str,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """
        Call an operation on a device, with argument types inferred from the
        JSONL 'method' specification.
        """
        key = (room_name, device_name, operation)
        if key not in self._methods_index:
            raise ValueError(f"Unsupported operation: {key}")

        spec = self._methods_index[key]
        device = self.get_device(room_name, device_name)

        # Convert kwargs to correct Python types based on spec
        typed_args = self._coerce_parameters(spec, kwargs)

        if not hasattr(device, operation):
            raise AttributeError(
                f"Device {device_name} in {room_name} has no method '{operation}'"
            )

        fn = getattr(device, operation)
        return fn(**typed_args)

    # ---------- Internal helpers ----------

    def _build_devices(self, home_status: Dict[str, Any]) -> None:
        for room_name, room_info in home_status.items():
            self.rooms[room_name] = {}
            # skip "room_name" field inside room_info
            for device_name, dev_info in room_info.items():
                if device_name == "room_name":
                    continue

                device_cls = self._map_device_name_to_class(device_name)
                if device_cls is None:
                    # Unknown device type; skip or store raw
                    continue

                device = self._instantiate_device(
                    device_cls=device_cls,
                    room_name=room_name,
                    device_name=device_name,
                    dev_info=dev_info,
                )
                self.rooms[room_name][device_name] = device

    def _map_device_name_to_class(self, device_name: str):
        """
        Map JSON device_name to the corresponding class in visual_home/device.py.
        Adjust this mapping to exactly match that file’s class names.
        """
        return device_map.get(device_name)

    def _instantiate_device(
        self,
        device_cls: Any,
        room_name: str,
        device_name: str,
        dev_info: Dict[str, Any],
    ) -> Any:
        """
        Instantiate a device and apply initial state/attributes.
        If the device classes in HomeBench provide their own init signature
        or setters, adjust this logic accordingly.
        """
        # Most HomeBench devices can be created with just (room_name, device_name)
        #print(device_cls)
        try:
            device = device_cls(state=dev_info['state'])
        except TypeError:
            # Fallback to a simpler signature
            try:
                device = device_cls(name=device_name)
            except TypeError:
                device = device_cls()

        # Apply state
        state = dev_info.get("state", None)
        attributes = dev_info.get("attributes", {})

        # Many HomeBench devices expose a .set_state or specific methods; here we
        # attempt a generic pattern. Adapt to the actual API in device.py.
        if state is not None and hasattr(device, "state"):
            device.state = state

        # Apply attributes as "best effort"
        self._apply_attributes(device, attributes)

        return device

    def _apply_attributes(self, device: Any, attributes: Dict[str, Any]) -> None:
        """
        Apply attributes from JSON to a device object in a best-effort way.
        For scalar 'value' fields, try to set device.<attr_name>.
        """
        for attr_name, attr_info in attributes.items():
            # Normalize attribute name: strip spaces for keys like " degree"
            clean_name = attr_name.strip()
            if isinstance(attr_info, dict) and "value" in attr_info:
                value = attr_info["value"]
            else:
                value = attr_info

            # Try direct attribute set
            if hasattr(device, clean_name):
                try:
                    setattr(device, clean_name, value)
                    continue
                except Exception:
                    pass

            # Try setter method: set_<attr>
            setter_name = f"set_{clean_name}"
            if hasattr(device, setter_name):
                try:
                    setter = getattr(device, setter_name)
                    setter(value)
                    continue
                except Exception:
                    pass
            # If neither works, silently ignore; adjust if stricter behavior needed

    def _coerce_parameters(
        self, spec: DeviceSpec, provided: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coerce provided kwargs into the types described in spec.parameters.
        Supported types:
          - int
          - str
          - typing.Tuple[int, int, int]
        """
        result: Dict[str, Any] = {}

        for param in spec.parameters:
            name = param["name"]
            type_str = param["type"]

            if name not in provided:
                raise ValueError(
                    f"Missing required parameter '{name}' for "
                    f"{spec.room_name}.{spec.device_name}.{spec.operation}"
                )

            raw_val = provided[name]

            if type_str == "int":
                result[name] = int(raw_val)
            elif type_str == "str":
                result[name] = str(raw_val)
            elif type_str.startswith("typing.Tuple[int, int, int]"):
                # Expect iterable of 3 ints
                if not isinstance(raw_val, (list, tuple)) or len(raw_val) != 3:
                    raise ValueError(
                        f"Parameter '{name}' must be a 3-element tuple/list of ints"
                    )
                result[name] = tuple(int(x) for x in raw_val)
            else:
                # Fallback: no conversion
                result[name] = raw_val

        return result


# ---------- JSONL Loader ----------

def load_home_from_jsonl(path: str) -> Dict:
    """
    Load a single-line JSONL file describing one home and
    return an initialized SmartHomeEnvManager.
    """
    # with open(path, "r", encoding="utf-8") as f:
    #     line = f.readline().strip()
    #     if not line:
    #         raise ValueError("Empty JSONL file")

    #     raw = json.loads(line)
    def load_homes(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    homes = {h["home_id"]: h for h in load_homes(path)}
    return homes

def load_home_from_id(home_id:int, homes: Dict)  -> SmartHomeEnvManager:
    raw = homes[home_id]

    # Parse methods
    methods = [
        DeviceSpec(
            room_name=m["room_name"],
            device_name=m["device_name"],
            operation=m["operation"],
            parameters=m.get("parameters", []),
        )
        for m in raw["method"]
    ]

    config = HomeConfig(
        home_id=raw["home_id"],
        methods=methods,
        home_status=raw["home_status"],
    )

    return SmartHomeEnvManager(config)


# ---------- Example usage ----------

if __name__ == "__main__":
    # Example: adjust path to your JSONL file
    homes = load_home_from_jsonl("/Users/yash/AgentX AgentBeats/AgentBeats_Manager/data/home_status_method_all.jsonl")
    manager = load_home_from_id(0, homes)
    # List rooms and devices
    print("Rooms:", manager.list_rooms())
    print("Devices in master_bedroom:", manager.list_devices("master_bedroom"))

    # Call operations
    manager.call(
        room_name="master_bedroom",
        device_name="air_conditioner",
        operation="set_temperature",
        temperature=24,
    )
    manager.call(
        room_name="living_room",
        device_name="light",
        operation="set_color",
        color=[255, 128, 64],
    )
    manager.call(
        room_name="garage",
        device_name="garage_door",
        operation="open",
    )

    # Inspect a device
    ac = manager.get_device("master_bedroom", "air_conditioner")
    print("Master bedroom AC state:", getattr(ac, "state", None))
