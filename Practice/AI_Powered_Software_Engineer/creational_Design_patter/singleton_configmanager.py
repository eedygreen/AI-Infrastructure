class ConfigManager:
    _instance = None

    def __new__(cls, *agrs, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'settings'):       # if there is an attribute of settings
            self.settings = {}

    def set(self, key, value):
        self.settings[key] = value

    def get(self, key):
        if key not in self.settings:
            raise KeyError(f"Setting '{key}' not found.")
        return self.settings[key]


import unittest

class TestingSingletonConfigManager(unittest.TestCase):
    def test_single_instance(self):
        config1 = ConfigManager()
        config2 = ConfigManager()

        self.assertIs(config1, config2)

    def test_Set_and_get(self):
        config = ConfigManager()
        config.set("theme", "dark")

        self.assertEqual(config.get("theme"), "dark")

    def test_share_state(self):
        config1 = ConfigManager()
        config1.set("Language", "Arabic")

        self.assertEqual(config1.get("Language"), "Arabic")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)