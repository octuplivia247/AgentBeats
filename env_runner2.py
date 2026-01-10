from src.green_agent.core.manager2 import *
# manager = load_home_from_jsonl("data/home_status_method.jsonl")

# # List rooms and devices
# print("Rooms:", manager.list_rooms())
# print("Devices in master_bedroom:", manager.list_devices("master_bedroom"))

# # Call operations
# manager.call(
#     room_name="master_bedroom",
#     device_name="air_conditioner",
#     operation="set_temperature",
#     temperature=24,
# )
# manager.call(
#     room_name="living_room",
#     device_name="light",
#     operation="set_color",
#     color=[255, 128, 64],
# )
# manager.call(
#     room_name="garage",
#     device_name="garage_door",
#     operation="open",
# )

# # Inspect a device
# ac = manager.get_device("master_bedroom", "air_conditioner")
# print("Master bedroom AC state:", getattr(ac, "state", None))

homes = load_home_from_jsonl("data/home_status_method_all.jsonl")
manager = load_home_from_id(32, homes)
# List rooms and devices
print("Rooms:", manager.list_rooms())
print("Devices in master_bedroom:", manager.list_devices("master_bedroom"))

# Call operations
manager.call(
    room_name="master_bedroom",
    device_name="heating",
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
ac = manager.get_device("master_bedroom", "aromatherapy")
print("Master bedroom AC state:", getattr(ac, "state", None))
