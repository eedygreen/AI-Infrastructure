# transaction.py

from transaction.transaction_category import TransactionCategory

class Transaction:
    """Represents a financial transaction with an amount and category."""

    def __init__(self, amount, category: TransactionCategory):
        self.amount = amount
        self.category = category

    def __str__(self):
        if isinstance(self.amount, int) or (isinstance(self.amount, float) and self.amount.is_integer()):
            amount_str = f"${int(self.amount)}"
        else:
            amount_str = f"${self.amount}"
        return f"Transaction({amount_str}, category='{self.category}')"

    def __eq__(self, other):
        if not isinstance(other, Transaction):
            return False
        return self.amount == other.amount and self.category == other.category
