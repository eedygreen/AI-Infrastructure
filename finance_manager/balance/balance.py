# balance.py

from transaction.transaction_category import TransactionCategory

class Balance:
    """Singleton to track the balance."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Balance, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the balance. Prevent direct instantiation."""
        if not self._initialized:
            self._balance = 0.0
            self._observers = []
            self._initialized = True

    @classmethod
    def get_instance(cls):
        """Ge the Signleton instance of Balance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register_observer(self, observer):
        """Register an observer to be notified of balance change"""
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister_observer(self, observer):
        """Unregistered an observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, transaction):
        """Notify all registered observers of a balance change"""
        for observer in self._observers:
            observer.update(self, transaction)

    def reset(self):
        """Reset the net balance to zero."""
        self._balance = 0.0

    def add_income(self, amount):
        """Add income to the balance."""
        self._balance += amount
    
    def add_expense(self, amount):
        """Subtract expense from the balance."""
        self._balance -= amount

    def apply_transaction(self, transaction):
        """
        Apply a Transaction object to update the balance.

        Args:
            transaction (Transaction): The transaction to apply.
        """
        if transaction.category == TransactionCategory.INCOME:
            self.add_income(transaction.amount)
        elif transaction.category == TransactionCategory.EXPENSE:
            self.add_expense(transaction.amount)
        else:
            raise ValueError(f"Invalid transaction category: {transaction.category}")

        self.notify_observers(transaction)

    def get_balance(self):
        """Get the current net balance."""
        if not hasattr(self, '_balance'):
            self._balance = 0.0
        return self._balance

    def summary(self):
        """Return a summary string of the net balance."""
        return f"Net Summary | Current Balance: | ${self.get_balance():.2f} |"
    
