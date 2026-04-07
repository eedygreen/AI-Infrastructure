class StockMarket:
    def __init__(self):
        self._investors = []
        self._price = None

    def subscribe(self, investor):
        if investor not in self._investors:
            self._investors.append(investor)

    def unsubscribe(self, investor):
        if investor in self._investors:
            self._investors.remove(investor)

    def set_price(self, price):
        print(f"\nStockMarket Price: New Price ${price}")
        self._price = price
        self.notify()

    def notify(self):
        for investor in self._investors:
            investor.update(self._price)


class Investor():
    def __init__(self, name):
        self.name = name
        self.notifications = []

    def update(self, price):
        message = f"{self.name} notified: Stock price is now {price}"
        self.notifications.append(message)
        print(message)

import unittest
class TestObserverPattern(unittest.TestCase):
    def test_notification_flow(self):
        market = StockMarket()
        investor1 = Investor("Alice")
        investor2 = Investor("Bob")

        market.subscribe(investor1)
        market.subscribe(investor2)

        market.set_price(100)

        self.assertEqual(investor1.notifications[0], "Alice notified: Stock price is now 100")
        self.assertEqual(investor2.notifications[0], "Bob notified: Stock price is now 100")

    def test_unsubscribe(self):
        market = StockMarket()
        investor = Investor("Charlie")

        market.subscribe(investor)
        market.unsubscribe(investor)
        market.set_price(200)

        self.assertEqual(len(investor.notifications), 0)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

