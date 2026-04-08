import unittest
from enum import Enum

class TransactionCategory(Enum):
    INCOME = "Income"
    EXPENSE  = "Expense"

class Transaction:
    def __init__(self, amount, category):
        self.amount = amount
        self.category = category

class MockBalance:
    def __init__(self):
        self._balance = 0

    def get_balance(self):
        return self._balance
    
    def set_balance(self, amount):
        self._balance = amount
    
    def apply_transaction(self, transaction):
        if transaction.category == TransactionCategory.INCOME:
            self._balance += transaction.amount
        elif transaction.category == TransactionCategory.EXPENSE:
            self._balance -= transaction.amount


from transaction.transaction_decorator import (
    LoggingDecorator,
    AuditDecorator,
    transaction_logger,
    transaction_validator,
    balance_checker,
    audit_trail,
    full_transaction_decorator
)

class TestLoggingDecorator(unittest.TestCase):

    def setUp(self):
        LoggingDecorator.clear_logs()
        self.balance = MockBalance()

        @transaction_logger
        def apply_transaction(balance, transaction):
            balance.apply_transaction(transaction)

        self.apply_transaction = apply_transaction

    def test_loggin_decorator_logs_transaction(self):
        """Test that logging decorator creates Log entries."""

        txn = Transaction(100, TransactionCategory.INCOME)
        self.apply_transaction(self.balance, txn)

        logs = LoggingDecorator.get_logs()
        self.assertEqual(len(logs), 2)
        self.assertIn("Applying Income: $100.00", logs[0])

    def test_logging_decorator_tracks_balance_changes(self):
        """Test that logging tracks balance changes"""

        self.balance.set_balance(500)
        txn = Transaction(100, TransactionCategory.EXPENSE)
        self.apply_transaction(self.balance, txn)

        logs = LoggingDecorator.get_logs()
        self.assertIn("$500.00 → $400.00", logs[1])

    def test_clear_logs(self):
        """Test clearing logs"""
        
        self.apply_transaction(self.balance, Transaction(100, TransactionCategory.INCOME))
        self.assertTrue(len(LoggingDecorator.get_logs()) > 0)

        LoggingDecorator.clear_logs()
        self.assertEqual(len(LoggingDecorator.get_logs()), 0)

class TestValidationDecorator(unittest.TestCase):

    def setUp(self):
        self.balance = MockBalance()

        @transaction_validator(min_amount=0.01, max_amount=10000)
        def apply_transaction(balance, transaction):
            balance.apply_transaction(transaction)
            return "success"
        
        self.apply_transaction = apply_transaction

    def test_validation_rejects_negative_amount(self):
        """Test validator rejects negative amounts."""

        txn = Transaction(-100, TransactionCategory.INCOME)

        with self.assertRaises(ValueError) as context:
            self.apply_transaction(self.balance, txn)

        self.assertIn("must be positive", str(context.exception))

    def test_validation_rejects_amount_below_minimum(self):
        """Validator rejects amounts below minimum."""

        @transaction_validator(min_amount=10, max_amount=10000)
        def apply_custom(balance, transaction):
            balance.apply_transaction(transaction)

        txn = Transaction(5, TransactionCategory.INCOME)

        with self.assertRaises(ValueError) as context:
            apply_custom(self.balance, txn)

        self.assertIn("below minimum", str(context.exception))

    def test_validation_rejects_amount_above_maximum(self):
        """Validator rejects amounts above maximum"""

        @transaction_validator(min_amount=0.01, max_amount=1000)
        def apply_custom(balance, transaction):
            balance.apply_transaction(transaction)

        txn = Transaction(5000, TransactionCategory.INCOME)

        with self.assertRaises(ValueError) as context:
            apply_custom(self.balance, txn)

        self.assertIn("exceeds maximum", str(context.exception))

    def test_validation_accepts_valid_amount(self):

        txn = Transaction(100, TransactionCategory.INCOME)
        result = self.apply_transaction(self.balance, txn)

        self.assertEqual(result, "success")
        self.assertEqual(self.balance.get_balance(), 100)

class TestBalanceCheckDecorator(unittest.TestCase):
    def setUp(self):
        self.balance = MockBalance()

        @balance_checker(allow_negative=False)
        def apply_transaction(balance, transaction):
            balance.apply_transaction(transaction)

        self.apply_transaction = apply_transaction

    def test_balance_check_prevents_overdraft(self):
        self.balance.set_balance = 100
        txn = Transaction(200, TransactionCategory.EXPENSE)

        with self.assertRaises(ValueError) as context:
            self.apply_transaction(self.balance, txn)

        self.assertIn("Insufficient funds", str(context.exception))

    def test_balance_check_allows_income_always(self):
        self.balance.set_balance = 0
        txn = Transaction(100, TransactionCategory.INCOME)

        self.apply_transaction(self.balance, txn)
        self.assertEqual(self.balance.get_balance(), 100)

    def test_balance_check_allows_negative_when_enabled(self):
        @balance_checker(allow_negative=True)
        def apply_with_negative(balance, transaction):
            balance.apply_transaction(transaction)

        self.balance.set_balance(50)
        txn = Transaction(100, TransactionCategory.EXPENSE)

        apply_with_negative(self.balance, txn)
        self.assertEqual(self.balance.get_balance(), -50)

class TestAuditDecorator(unittest.TestCase):

    def setUp(self):
        AuditDecorator.clear_audit_log()
        self.balance = MockBalance()

        @audit_trail
        def apply_transaction(balance, transaction):
            balance.apply_transaction(transaction)

        self.apply_transaction = apply_transaction

    def test_audit_decorator_creates_audit_trail(self):
        txn = Transaction(100, TransactionCategory.INCOME)
        self.apply_transaction(self.balance, txn)

        audit_log = AuditDecorator.get_audit_log()
        self.assertEqual(len(audit_log), 1)

        entry = audit_log[0]
        self.assertEqual(entry["transaction_type"], "Income")
        self.assertEqual(entry["amount"], 100)
        self.assertEqual(entry["balance_before"], 0)
        self.assertEqual(entry["balance_after"], 100)
        self.assertEqual(entry["status"], "SUCCESS")

    def test_audit_log_tracks_multiple_transactions(self):
        self.apply_transaction(self.balance, Transaction(100, TransactionCategory.INCOME))
        self.apply_transaction(self.balance, Transaction(30, TransactionCategory.EXPENSE))
        
        audit_log = AuditDecorator.get_audit_log()
        self.assertEqual(len(audit_log), 2)

class TestCompositeDecorator(unittest.TestCase):
    def setUp(self):
        LoggingDecorator.clear_logs()
        AuditDecorator.clear_audit_log()
        self.balance = MockBalance()

        @full_transaction_decorator(min_amount=1, max_amount=1000, allow_negative=False)
        def apply_transaction(balance, transaction):
            balance.apply_transaction(transaction)

        self.apply_transaction = apply_transaction

        def test_full_decorator_applies_all_checks(self):
            txn = Transaction(100, TransactionCategory.INCOME)
            self.apply_transaction(self.balance, txn)
            
            self.assertTrue(len(LoggingDecorator.get_logs()) > 0)
            self.assertTrue(len(AuditDecorator.get_audit_log()) > 0)
            
            invalid_txn = Transaction(5000, TransactionCategory.INCOME)
            with self.assertRaises(ValueError):
                self.apply_transaction(self.balance, invalid_txn)

if __name__ == "__main__":
    unittest.main(verbosity=2)