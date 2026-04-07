from abc import ABC, abstractmethod

class Coffee(ABC):
    @abstractmethod
    def get_cost(self):
        pass

    @abstractmethod
    def get_ingredients(self):
        pass

# Concrete Component
class SimpleCoffee(Coffee):
    def get_cost(self):
        return 5

    def get_ingredients(self):
        return "Coffee"

# Decorator
class CoffeeDecorator(Coffee):
    def __init__(self, component):
        self.component = component
    
    @abstractmethod
    def get_cost(self):
        pass
    
    @abstractmethod
    def get_ingredients(self):
        pass

# Concrete Decorator
class MilkDecorator(CoffeeDecorator):
    def get_cost(self):
        return self.component.get_cost() + 2

    def get_ingredients(self):
        return self.component.get_ingredients() + ", Milk"

class SugarDecorator(CoffeeDecorator):
    def get_cost(self):
        return self.component.get_cost() + 1

    def get_ingredients(self):
        return self.component.get_ingredients() + ", Sugar"

class WhipDecorator(CoffeeDecorator):
    def get_cost(self):
        return self.component.get_cost() + 3

    def get_ingredients(self):
        return self.component.get_ingredients() + ", Whip"



if __name__ == "__main__":
    coffee = SimpleCoffee()
    coffee = MilkDecorator(coffee)
    coffee = SugarDecorator(coffee)
    coffee = WhipDecorator(coffee)

    print(coffee.get_cost())
    print(coffee.get_ingredients())

import unittest

class TestDecoratorPattern(unittest.TestCase):
    def tesst_simple_coffee(self):
        coffee = SimpleCoffee()
        self.assertEqual(coffee.get_cost(), 5)
        self.assertEqual(coffee.get_ingredients(), 'Coffee')

    def test_milk_and_sugar(self):
        coffee = MilkDecorator(SugarDecorator(SimpleCoffee()))
        self.assertEqual(coffee.get_ingredients(), "Coffee, Sugar, Milk")
        self.assertEqual(coffee.get_cost(), 5 + 2 + 1)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
