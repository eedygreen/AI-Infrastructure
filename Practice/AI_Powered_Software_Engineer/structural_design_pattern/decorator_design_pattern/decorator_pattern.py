
from abc import ABC, abstractmethod

# Component
class Product(ABC):
    @abstractmethod
    def get_price(self):
        pass

# Concrete Component
class BaseProduct(Product):
    def __init__(self, name, price):
        self.name = name
        self._price = price

    def get_price(self):
        return self._price

    
# Decorator
class ProductDecorator(Product):
    def __init__(self, product: Product):
        self._product = product

    def get_price(self):
        return self._product.get_price()
  
# Concrete Decorators
class SeasonalDiscount(ProductDecorator):
    def get_price(self):
        price = super().get_price()
        print("[Applying seasonal discount: 10% off]")
        return price * 0.9

class MemberDiscount(ProductDecorator):
    def get_price(self):
        price = super().get_price()
        print("[Applying member discount: 5% off]")
        return price * 0.95

class ShippingFee(ProductDecorator):
    def get_price(self):
        price = super().get_price()
        print("[Adding shipping fee: $10]")
        return price + 10

if __name__ == "__main__":
    product = BaseProduct("Laptop", 1000)

    print("Base Price: ", product.get_price())
    print()

    # Applying seasonal discount
    discounted = SeasonalDiscount(product)
    print("Price with seasonal discount: ", discounted.get_price())
    print()

    full_price = ShippingFee(MemberDiscount(SeasonalDiscount(product)))
    print("Price with Seasonl and Member discount: ", full_price.get_price())