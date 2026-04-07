
class WeatherStation:
    def __init__(self):
        self._observers = []
        self._temperature = None

    def subscribe(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def set_temperature(self, temperature):
        print(f"\nWeatherStation: New temperature = {temperature}°C")
        self._temperature = temperature
        self.notify()

    def notify(self):
        for observer in self._observers:
            observer.update(self._temperature)

# Observer Interface
class Observer:
    def update(self, temperature):
        raise NotImplementedError

# Concrete observers
class PhoneDisplay(Observer):
    def update(self, temperature):
        print(f"Phone Display: Temperature is {temperature}°C")

class TVDisplay(Observer):
    def update(self, temperature):
        print(f"TV Display: Temperature is {temperature}°C")

class SmartSpeaker(Observer):
    def update(self, temperature):
        print(f"Smart Speaker: Current temperature is {temperature}°C")

if __name__ == "__main__":
    # Create subject (Weather Station)
    station = WeatherStation()

    # Create observers
    phone = PhoneDisplay()
    tv = TVDisplay()
    speaker = SmartSpeaker()

    # Subscribe Observers
    station.subscribe(phone)
    station.subscribe(tv)
    station.subscribe(speaker)

    # Change Temperature  (Call observers get notified)
    station.set_temperature(25)
    station.set_temperature(30)

    # Unsubscribe TV and notify again
    station.unsubscribe(tv)
    station.set_temperature(20)
