
from devices import *
import json
import random
import inspect

AllCandiateRoom = ["master bedroom","guest bedroom","living room","ding room","study room","kitchen","bathroom","foyer","corridor","balcony","garage","store room"]
LightCandiateRoom = AllCandiateRoom

def generate_instructions():
    instructions = []
    l = LightDevice("on")
    for room in LightCandiateRoom:
        light_instruction = l.generate_instructions()
        for instr in light_instruction:
            instr["room"] = room
            instructions.append(instr)

    print(len(instructions))
    return instructions


def generate_subclass(base_class,attributes, operations):
    class SubClass(base_class):
        def __init__(self, state: str):
            super().__init__(state)
            self.attributes = {attr: self.attributes[attr] for attr in attributes if attr in self.attributes}
            self.operations = {op: self.operations[op] for op in operations if op in self.operations}
    return SubClass


class VisualMasterBedroom:
    def __init__(self) -> None:
        self.name = "master_bedroom"
        self.devices = []
        self.unexist_devices = []
        self.devices.append(random.choice(LightDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(AirConditionerDeviceList)("on"))
        #     self.unexist_devices.append(random.choice(HeatingDeviceList)("on"))
        #     self.unexist_devices.append(random.choice(FanDeviceList)("on"))
        # else:
        #     self.devices.append(random.choice(HeatingDeviceList)("on"))
        #     self.devices.append(random.choice(FanDeviceList)("on"))
        #     self.unexist_devices.append(random.choice(AirConditionerDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(CurtainDeviceList)("on"))
        # else:
        #     self.unexist_devices.append(random.choice(CurtainDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(AirPurifiersDeviceList)("on"))
        # else:
        #     self.unexist_devices.append(random.choice(AirPurifiersDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(HumidifierDeviceList)("on"))
        #     self.unexist_devices.append(random.choice(DehumidifiersDeviceList)("on"))
        # else:
        #     self.devices.append(random.choice(DehumidifiersDeviceList)("on"))
        #     self.unexist_devices.append(random.choice(HumidifierDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(AromatherapyDeviceList)("on"))
        # else:
        #     self.unexist_devices.append(random.choice(AromatherapyDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(TrashDeviceList)("on"))
        # else:
        #     self.unexist_devices.append(random.choice(TrashDeviceList)("on"))
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(MediaPlayerDeviceList)("play"))
        # else:
        #     self.unexist_devices.append(random.choice(MediaPlayerDeviceList)("play"))

        # if random.random() > 0.5:
        #     self.devices.append(BedDevice())
        # else:
        #     self.unexist_devices.append(BedDevice())
        
        # if random.random() > 0.5:
        #     self.devices.append(random.choice(PetFeederDeviceList)("on"))
        # else:
        #     self.unexist_devices.append(random.choice(PetFeederDeviceList)("on"))


        self.random_initialize()
        self.state = self.get_status()
        self.devices_name_list = [device.name for device in self.devices]


    def random_initialize(self):
        for device in self.devices:
            device.random_initialize()

    def get_status(self):
        state = {"room_name": self.name}
        for device in self.devices:
            state[device.name] = device.get_status()
        return state    
    
    def execute_instructions(self, instructions):
        for instr in instructions:
            if instr["room"] == self.name:
                if instr["device"] in self.devices_name_list:
                    device = self.devices[self.devices_name_list.index(instr["device"])]
                    if instr["instruction"] in device.operations.keys():
                        device.operations[instr["instruction"]]()

        state = self.get_status()

        return state
    
    def initalzie(self, room_state,methods):
        self.devices = []
        self.devices_name_list = []
        for device in room_state.keys():
            if device == "room_name":
                continue
            else:
                attributes = room_state[device]["attributes"].keys()
                new_device = generate_subclass(device_map[device],attributes,methods[device])
                self.devices.append(new_device(room_state[device]["state"]))
                self.devices_name_list.append(device)
                self.devices[-1].initialize(room_state[device]["state"],room_state[device]["attributes"])
        
        self.state = self.get_status()

        return self.state
    
class VisualHome:
    def __init__(self) -> None:
        self.rooms = []
        self.rooms.append(VisualMasterBedroom())
        # self.rooms.append(VisualGuestBedroom())
        # self.rooms.append(VisualLivingRoom())
        # self.rooms.append(VisualDingRoom())
        # self.rooms.append(VisualStudyRoom())
        # self.rooms.append(VisualKitchen())
        # self.rooms.append(VisualBathroom())
        # self.rooms.append(VisualFoyer())
        # self.rooms.append(VisualCorridor())
        # self.rooms.append(VisualBalcony())
        # self.rooms.append(VisualGarage())
        # self.rooms.append(VisualStoreRoom())
        if random.random() > 0.5:
            self.VacuumRobot = VacuumRobotrDevice("on")
        self.state = self.get_status()
        self.rooms_name_list = [room.name for room in self.rooms]

    def get_status(self):
        state = {}
        for room in self.rooms:
            state[room.name] = room.get_status()

        if hasattr(self, "VacuumRobot"):
            state["VacuumRobot"] = self.VacuumRobot.get_status()
        return state

    def execute_instructions(self, instructions):
        for instr in instructions:
            if instr["room"] in self.rooms_name_list:
                room = self.rooms[self.rooms_name_list.index(instr["room"])]
                room.execute_instructions([instr])
            elif instr["room"] == "VacuumRobot":
                if hasattr(self, "VacuumRobot"):
                    if instr["instruction"] in self.VacuumRobot.operations.keys():
                        self.VacuumRobot.operations[instr["instruction"]]()

        state = self.get_status()

        return state
    
    def initalzie(self, home_state):
        methods = {}
        methods_list = home_state["method"]
        home_status = home_state["home_status"]
        for method in methods_list:
            if method["room_name"] in methods.keys():
                if method["device_name"] in methods[method["room_name"]].keys():
                    methods[method["room_name"]][method["device_name"]].append(method["operation"])
                else:
                    methods[method["room_name"]][method["device_name"]] = [method["operation"]]
            else:
                methods[method["room_name"]] = {method["device_name"]: [method["operation"]]}
        for room in self.rooms:
            room_state = home_status[room.name]
            room.initalzie(room_state,methods[room.name])
        if "vacuum_robot" in home_status.keys():
            self.VacuumRobot.initialize(home_status["vacuum_robot"]["state"],home_status["vacuum_robot"]["attributes"])

        self.state = self.get_status()

        return self.state
    