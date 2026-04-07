from abc import ABC, abstractmethod

# Step !: Define dependency interface
class WeatherProvider(ABC):
    @bastracmethod
    def get_weather(self, location: str):
        pass

# Step 2: Concrete Implementations
class APIWeatherProvider(WeatherProvider):
    def get_weather(self, location: str):
        return f"API weather data for {location}"

class FileWeatherProvider(WeatherProvider):
    def get_weather(self, location: str):
        # In reality, this would read from a file
        return f"File-based weather data for {location}"

class MockWeartherProvider(WeatherProvider):
    def get_weather(self, location: str):
        return f"Mock weather: Always sunny in {location}"

# Step 3: Service that uses dependency injection
class WeatherService:
    def __init__(self, provider: WeatherProvider):
        self.provider = provider

    def report(self, location: str):
        return self.provider.get_weather(location)

if __name__ == "__main__":
    # Inejct APIWeatherProvider
    api_Service = WeatherService(APIWeatherProvider())
    print(api_service.report("Toronto"))


    # Inject FileWeatherProvider
    file_service = WeatherService(FileWeatherProvider())
    print(file_service.report("Toronto"))

    # Inject MockWeatherProvider
    mock_service = WeatherService(MockWeatherProvider())
    print(mock_service.report(("Toronto"))