from abc import ABC, abstractmethod

    # === Define interface ===
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class ShippingStrategy(ABC):
    @abstractmethod
    def ship(self, ship):
        pass

class NotificationService(ABC):
    @abstractmethod
    def notify(self, message):
        pass

    # === Concrete Implementation ===
# Payment
class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid {amount} using Credit Card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid {amount} using PayPal"

# Shiping
class StandardShipping(ShippingStrategy):
    def ship(self, order_id):
        return f"Shipped {order_id} via Standard"

class ExpressShipping(ShippingStrategy):
    def ship(self, order_id):
        return f"Shipped {order_id} via Express"

# Notification
class EmailNotification(NotificationService):
    def notify(self, message):
        return f"Notified via Email"

class SMSNotification(NotificationService):
    def notify(self, message):
        return f"Notified via SMS"

    # === Checkout Service
class CheckoutService:
    def __init__(self, payment: PaymentStrategy,
        shipping: ShippingStrategy,
        notification: NotificationService):

        self.payment = payment
        self.shipping = shipping
        self.notification = notification

    def checkout(self, order_id, amount):
        payment_result = self.payment.pay(amount)
        shipping_result = self.shipping.ship(order_id)
        notification_result = self.notification.notify(order_id)
        return f"{payment_result} | {shipping_result} | {notification_result}"


import unittest

class TestCheckoutDI(unittest.TestCase):
    def test_creditcard_standard_email(self):
        service = CheckoutService(
            payment=CreditCardPayment(),
            shipping=StandardShipping(),
            notification=EmailNotification()
        )
        self.assertEqual(
            service.checkout("ORDER123", 100),
            "Paid 100 using Credit Card | Shipped ORDER123 via Standard | Notified via Email"
        )

    def test_paypal_express_sms(self):
        service = CheckoutService(
            payment=PayPalPayment(),
            shipping=ExpressShipping(),
            notification=SMSNotification()
        )
        self.assertEqual(
            service.checkout("ORDER456", 200),
            "Paid 200 using PayPal | Shipped ORDER456 via Express | Notified via SMS"
        )

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)