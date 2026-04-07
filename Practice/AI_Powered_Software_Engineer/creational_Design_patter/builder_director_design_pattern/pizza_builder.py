
class Pizza:
    def __init__(self):
        self.toppings = []

    def __str__(self):
        return f"Pizza with toppings: {','.join(self.toppings) if self.toppings else 'no toppings'}"
    
class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()

    def add_cheese(self):
        self.pizza.toppings.append("cheese")
        return self
    
    def add_mushrooms(self):
        self.pizza.toppings.append("mushrooms")
        return self
    
    def add_olives(self):
        self.pizza.toppings.append("olives")
        return self
    
    def build(self):
        return self.pizza
    
if __name__ == "__main__":
    builder = PizzaBuilder()
    pizza1 = (
        builder.add_cheese().add_mushrooms().build()
    )
    print(pizza1)

    builder2 = PizzaBuilder()
    pizza2 = builder2.add_cheese().add_olives().build()
    print(pizza2)

    plain_pizza = PizzaBuilder().build()
    print(plain_pizza)