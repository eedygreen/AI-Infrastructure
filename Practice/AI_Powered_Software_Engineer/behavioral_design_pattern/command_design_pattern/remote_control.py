from abc import ABC, abstractmethod

# step 1: Command Interface
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

# Step 2: Receivers
class Light:
    def turn_on(self):
        return "Light is ON"

    def turn_off(self):
        return "Light is OFF"

class Fan:
    def turn_on(self):
        return "Fan is ON"
    
    def turn_off(self):
        return "Fan is OFF"

# Step 3: Concrete Commands
class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        return self.light.turn_on()
    
class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light

    def execute(self):
        return self.light.turn_off()

class FanOnCommand(Command):
    def __init__(self, fan: Fan):
        self.fan = fan

    def execute(self):
        return self.fan.turn_on()

class FanOffCommand(Command):
    def __init__(self, fan: Fan):
        self.fan = fan

    def execute(self):
        return self.fan.turn_off()

# Step 4: Invoker
class RemoteControl:
    def __init__(self):
        self._command = None

    def set_command(self, command: Command):
        self._command = command

    def press_button(self):
        if self._command:
            return self._command.execute()
        return "No command set"

import unittest

class TestCommandPattern(unittest.TestCase):
    def test_light_on(self):
        remote = RemoteControl()
        light = Light()
        remote.set_command(LightOnCommand(light))
        self.assertEqual(remote.press_button(), "Light is ON")

    def test_light_off(self):
        remote = RemoteControl()
        light = Light()
        remote.set_command(LightOffCommand(light))
        self.assertEqual(remote.press_button(), "Light is OFF")

    def test_fan_on(self):
        remote = RemoteControl
        fan = Fan()
        remote.set_command(FanOnCommand(fan))
        self.assertEqual(remote.press_buttong(), "Fan is ON")

    def test_fan_off(self):
        remote = RemoteControl()
        fan = Fan()
        remote.set_command(FanOffCommand(fan))
        self.assertEqual(remote.press_button(), "Fan is OFF")

    def test_no_command(self):
        remote = RemoteControl()
        self.assertEqual(remote.press_button(), "No command set")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)