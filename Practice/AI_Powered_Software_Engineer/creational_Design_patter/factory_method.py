
class Shape:
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        return "Drawing Circle"
    
class Square(Shape):
    def draw(self):
        return "Drawing Square"

class Triangle(Shape):
    def draw(self):
        return "Drawing Triangle"
    
class ShapeFactory:
    @staticmethod
    def get_shape(shape_type):
        shape_type = shape_type.lower()
        
        if shape_type == "circle":
            return Circle()
        elif shape_type == "square":
            return Square()
        elif shape_type == "triangle":
            return Triangle()
        else:
            raise ValueError("Invalid shape type")
        

if __name__ == "__main__":
    shapes = ["circle", "square", "triangle"]

    for s in shapes:
        shape = ShapeFactory.get_shape(s)
        if shape:
            print(shape.draw())
        else:
            print(f"No shape for found for: {s}")

import unittest

class TestShapeFactory(unittest.TestCase):
    def setUp(self):
        self.factory = ShapeFactory()

    def test_circle(self):
        shape = self.factory.get_shape("circle")
        self.assertEqual(shape.draw(), "Drawing Circle")

    def test_square(self):
        shape = self.factory.get_shape("square")
        self.assertEqual(shape.draw(), "Drawing Square")

    def test_triangle(self):
        shape = self.factory.get_shape("triangle")
        self.assertEqual(shape.draw(), "Drawing Triangle")
    
    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            self.factory.get_shape("hexagon")
            
if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)