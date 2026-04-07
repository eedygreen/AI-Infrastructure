
class House:
    def __init__(self):
        self.walls = None
        self.roof = None
        self.windows = None
        self.doors = None
        self.garage = None

    def show_details(self):
        return (f"Walls: {self.walls}, Roofs: {self.roof}, Windows: {self.windows}, "
                f"Doors: {self.doors}, Garage: self.{self.garage}")
    
class HouseBuilder:
    def __init__(self):
        self.house = House()

    def set_walls(self, wall_type):
        self.house.walls = wall_type
        return self
    
    def set_roof(self, roof_type):
        self.house.roof = roof_type
        return self
    
    def set_windows(self, count):
        self.house.windows = count
        return self
    
    def set_doors(self, count):
        self.house.doors = count
        return self

    def set_garage(self, garage_type):
        self.house.garage = garage_type
        return self
    
    def build(self):
        return self.house
    
if __name__ == "__main__":
    builder = HouseBuilder()
    house1 = builder.set_walls("Marble").set_roof("Slate").set_doors(7).set_windows(6).build()
    print("Beautiful House:", house1.show_details())

    builder2 = HouseBuilder()
    house2 = builder2.set_walls("Blocks").set_roof("Flat").set_doors(8).set_windows(13).set_garage("Four Car Parks").build()
    print("Beautiful House with Garden: ", house2.show_details())


import unittest

class TestHouseBuilder(unittest.TestCase):

    def test_build_house(self):
        builder = HouseBuilder()
        house = builder.set_walls("Brick").set_roof("Tile").set_windows(4).build()
        self.assertEqual(house.show_details(),
                         "Walls: Brick, Roof: Tile, Windows: 4, Doors: None, Garage: None")

    def test_chaining_all_components(self):
        builder = HouseBuilder()
        house = (builder.set_walls("Concrete")
                     .set_roof("Shingle")
                     .set_windows(6)
                     .set_doors(2)
                     .set_garage("Double")
                     .build())
        self.assertEqual(house.show_details(),
                         "Walls: Concrete, Roof: Shingle, Windows: 6, Doors: 2, Garage: Double")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)