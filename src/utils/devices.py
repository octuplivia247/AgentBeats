#all device implementations from HomeBench
#todo: should we import homebench as a submodel or copy over a few files?

from typing import Any, Dict, List, Callable, Optional, Tuple
import random

AllCandiateRoom = [
    "master bedroom",
    "guest bedroom",
    "living room",
    "ding room",
    "study room",
    "kitchen",
    "bathroom",
    "foyer",
    "corridor",
    "balcony",
    "garage",
    "store room"
]

class LightDevice:
    def __init__(self, state: str):
        self.name = "light"
        self.state = state  # e.g., "on", "off"
        self.attributes = {
            "brightness": {"value": 100, "lowest": "0", "highest": "100"},
            "color": {"value": (255,255,255)},
        }
        self.operations = {
            "turn_on": self.turn_on,
            "turn_off": self.turn_off,
            "set_brightness": self.set_brightness,
            "set_color": self.set_color
        }

    def turn_on(self):
        self.state = "on"
        print("Device is now ON.")

    def turn_off(self):
        self.state = "off"
        print("Device is now OFF.")
    
    def set_brightness(self, brightness: int):
        if self.state == "off":
            print("Cannot set brightness while device is off.")
        else:
            self.attributes["brightness"]["value"] = brightness
            print(f"Brightness set to {brightness}.")
    
    def set_color(self, color: Tuple[int, int, int]):
        if self.state == "off":
            print("Cannot set color while device is off.")
        else:
            self.attributes["color"]["value"] = color
            print(f"Color set to {color}.")

    def initialize(self, state: str, attributes: Optional[Dict[str, Any]] = None):
        self.state = state
        if attributes:
            self.attributes = attributes

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "attributes": self.attributes
        }
    
    def random_initialize(self):
        self.state = random.choice(["on", "off"])
        self.attributes["brightness"]["value"] = random.randint(0, 100)
        self.attributes["color"]["value"] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return self.get_status()

    def generate_instructions(self):
        instructions_list = []
        instructions_list.append({"instruction": "turn_on()", "device": self.name})
        instructions_list.append({"instruction": "turn_off()", "device": self.name})
        if "brightness" in self.attributes.keys():
            for brightness in range(0, 101, 10):
                instructions_list.append({"instruction": f"set_brightness({brightness})", "device": self.name})

                if self.attributes["brightness"]["value"] < brightness:
                    instructions_list.append({"instruction": f"set_brightness({brightness})", "device": self.name, "explain": "increase " + str(brightness-self.attributes["brightness"]["value"]) + " precent"})
                elif self.attributes["brightness"]["value"] > brightness:
                    instructions_list.append({"instruction": f"set_brightness({brightness})", "device": self.name, "explain": "decrease " + str(self.attributes["brightness"]["value"]-brightness) + " precent"})

        return instructions_list
    
    def generate_unexist_instructions(self):
        instructions_list = []
        for brightness in range(0, 101, 10):
                instructions_list.append({"instruction": f"set_brightness({brightness})", "device": self.name})
        return instructions_list
    