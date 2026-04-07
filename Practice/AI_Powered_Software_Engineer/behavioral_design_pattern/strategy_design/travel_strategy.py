from abc import ABC, abstractmethod

# Strategy Interface
class TravelStrategy(ABC):
    @abstractmethod
    def travel(self, start, end):
        pass

# Concrete Strategies
class CarTravel(TravelStrategy):
    def travel(self, start, end):
        return f"Driving from {start} to {end}"
    
class BikeTravel(TravelStrategy):
    def travel(self, start, end):
        return f"Cycling from {start} to {end}"

class WalkingTravel(TravelStrategy):
    def travel(self, start, end):
        return f"Walking from {start} to {end}"

# Context
class TravelPlanner:
    def __init__(self, strategy: TravelStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TravelStrategy):
        self._strategy = strategy

    def plan_route(self, start, end):
        return self._strategy.travel(start, end)

if __name__ == "__main__":
    planner = TravelPlanner(CarTravel()) # start with car
    print(planner.plan_route("Home", "Office"))

    # Swicth to Bike
    planner.set_strategy(BikeTravel())
    print(planner.plan_route("Home", "Park"))

    # Switch to walking
    planner.set_strategy(WalkingTravel())
    print(planner.plan_route("Cafe", "Library"))