from abc import ABC, abstractmethod

# Component
class FileSystemComponent(ABC):
    @abstractmethod
    def show_details(self, indent=0):
        pass

    @abstractmethod
    def get_size(self):
        pass

# Leaf
class File(FileSystemComponent):
    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def show_details(self, indent=0):
        print(" " * indent + f"{self.name}: {self.size}")

    def get_size(self):
        return self.size


# composite
class Directory(FileSystemComponent):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add(self, component):
        return self.children.append(component)

    def show_details(self, indent=0):
        print(" " * indent + f"Directory: {self.name}")
        for child in self.children:
            child.show_details(indent + 2)

    def get_size(self):
        return sum(child.get_size() for child in self.children)

if __name__ == "__main__":
    root = Directory("root")
    file1 = File("file1.txt", 100)
    file2 = File("fil2e.txt", 200)

    subdir = Directory("subdir")
    subdir.add(File("file3.txt", 300))

    root.add(file1)
    root.add(file2)
    root.add(subdir)

    root.show_details()
    print("Total size: ", root.get_size())

import unittest

class TestCompositePattern(unittest.TestCase):
    def test_file_size(self):
        file = File("test.txt", 150)
        self.assertEqual(file.get_size(), 150)

    def test_directory_size(self):
        root = Directory("root")
        root.add(File("a.txt", 100))
        root.add(File("b.txt", 200))
        self.assertEqual(root.get_size(), 300)

    def test_nested_directory(self):
        root = Directory("root")
        subdir = Directory("subdir")
        subdir.add(File("c.txt", 300))
        root.add(subdir)
        self.assertEqual(root.get_size(), 300)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
