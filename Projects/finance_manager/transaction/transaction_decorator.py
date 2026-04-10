from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Callable
 
 
class TransactionDecorator(ABC):
    """
    Abstract base decorator for transaction operations.
    Implements the Decorator Pattern to add validation and logging capabilities.
    """
    
    def __init__(self, wrapped_function: Callable = None):
        """
        Initialize the decorator.
        
        Args:
            wrapped_function: The function to wrap (for function decorator usage)
        """
        self.wrapped_function = wrapped_function
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Execute the decorated function with additional behavior."""
        pass
 
class LoggingDecorator(TransactionDecorator):
    """
    Decorator that logs all transaction operations.
    Records timestamp, transaction details, and balance changes.
    """
    
    # Class-level log storage
    _logs: List[str] = []
    
    def __init__(self, wrapped_function: Callable = None):
        super().__init__(wrapped_function)
    
    def __call__(self, *args, **kwargs):
        """Log the transaction before and after execution."""
        # Get transaction info
        if len(args) > 1:
            transaction = args[1]  # Assuming (self, transaction) signature
            balance = args[0]
            
            # Get balance before
            balance_before = balance.get_balance()
            
            # Log before execution
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] Applying {transaction.category.value}: ${transaction.amount:.2f}"
            self._logs.append(log_entry)
            
            # Execute the wrapped function
            result = self.wrapped_function(*args, **kwargs)
            
            # Log after execution
            balance_after = balance.get_balance()
            log_entry = f"[{timestamp}] Balance: ${balance_before:.2f} → ${balance_after:.2f}"
            self._logs.append(log_entry)
            
            return result
        else:
            # No transaction argument, just execute
            return self.wrapped_function(*args, **kwargs)
    
    @classmethod
    def get_logs(cls) -> List[str]:
        """Get all logged transactions."""
        return cls._logs.copy()
    
    @classmethod
    def clear_logs(cls):
        """Clear all logged transactions."""
        cls._logs.clear()
    
    @classmethod
    def print_logs(cls):
        """Print all logged transactions."""
        for log in cls._logs:
            print(log)

class ValidationDecorator(TransactionDecorator):
    """
    Decorator that validates transactions before execution.
    Ensures transactions meet business rules.
    """
    
    def __init__(self, wrapped_function: Callable = None, min_amount: float = 0.01, max_amount: float = 1000000):
        """
        Initialize validator with constraints.
        
        Args:
            wrapped_function: The function to wrap
            min_amount: Minimum transaction amount
            max_amount: Maximum transaction amount
        """
        super().__init__(wrapped_function)
        self.min_amount = min_amount
        self.max_amount = max_amount
    
    def __call__(self, *args, **kwargs):
        """Validate the transaction before execution."""
        if len(args) > 1:
            transaction = args[1]
            
            # Validate amount is positive
            if transaction.amount <= 0:
                raise ValueError(f"Transaction amount must be positive, got {transaction.amount}")
            
            # Validate amount is within range
            if transaction.amount < self.min_amount:
                raise ValueError(f"Transaction amount ${transaction.amount:.2f} is below minimum ${self.min_amount:.2f}")
            
            if transaction.amount > self.max_amount:
                raise ValueError(f"Transaction amount ${transaction.amount:.2f} exceeds maximum ${self.max_amount:.2f}")
            
            # Validate transaction has required attributes
            if not hasattr(transaction, 'amount'):
                raise ValueError("Transaction must have 'amount' attribute")
            
            if not hasattr(transaction, 'category'):
                raise ValueError("Transaction must have 'category' attribute")
        
        # Execute the wrapped function
        return self.wrapped_function(*args, **kwargs)


class BalanceCheckDecorator(TransactionDecorator):
    """
    Decorator that prevents transactions that would result in negative balance.
    Implements overdraft protection.
    """
    
    def __init__(self, wrapped_function: Callable = None, allow_negative: bool = False):
        """
        Initialize balance checker.
        
        Args:
            wrapped_function: The function to wrap
            allow_negative: Whether to allow negative balances
        """
        super().__init__(wrapped_function)
        self.allow_negative = allow_negative
    
    def __call__(self, *args, **kwargs):
        """Check if transaction would cause negative balance."""
        if len(args) > 1 and not self.allow_negative:
            transaction = args[1]
            balance = args[0]
            
            # Check if this is an expense
            if hasattr(transaction, 'category'):
                # Check category value instead of importing
                category_value = str(transaction.category.value) if hasattr(transaction.category, 'value') else str(transaction.category)
                if category_value == "Expense":
                    current_balance = balance.get_balance()
                    new_balance = current_balance - transaction.amount
                    
                    if new_balance < 0:
                        raise ValueError(
                            f"Insufficient funds: Balance ${current_balance:.2f} - "
                            f"Expense ${transaction.amount:.2f} = ${new_balance:.2f}"
                        )
        
        # Execute the wrapped function
        return self.wrapped_function(*args, **kwargs)
    
class AuditDecorator(TransactionDecorator):
    """
    Decorator that creates detailed audit trail for compliance.
    Records all transaction details for regulatory purposes.
    """
    
    # Class-level audit log
    _audit_log: List[dict] = []
    
    def __init__(self, wrapped_function: Callable = None):
        super().__init__(wrapped_function)
    
    def __call__(self, *args, **kwargs):
        """Create audit entry for the transaction."""
        if len(args) > 1:
            transaction = args[1]
            balance = args[0]
            
            # Create audit entry
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "transaction_type": transaction.category.value if hasattr(transaction, 'category') else 'Unknown',
                "amount": transaction.amount if hasattr(transaction, 'amount') else 0,
                "balance_before": balance.get_balance(),
            }
            
            # Execute transaction
            result = self.wrapped_function(*args, **kwargs)
            
            # Complete audit entry
            audit_entry["balance_after"] = balance.get_balance()
            audit_entry["status"] = "SUCCESS"
            
            self._audit_log.append(audit_entry)
            
            return result
        else:
            return self.wrapped_function(*args, **kwargs)
    
    @classmethod
    def get_audit_log(cls) -> List[dict]:
        """Get complete audit log."""
        return cls._audit_log.copy()
    
    @classmethod
    def clear_audit_log(cls):
        """Clear audit log."""
        cls._audit_log.clear()


def transaction_logger(func):
    """
    Function decorator for logging transactions.
    Usage: @transaction_logger
    """
    decorator = LoggingDecorator(func)
    return decorator

def transaction_validator(min_amount: float = 0.01, max_amount: float = 1000000):
    """
    Function decorator factory for validating transactions.
    Usage: @transaction_validator(min_amount=1, max_amount=10000)
    """
    def decorator(func):
        return ValidationDecorator(func, min_amount, max_amount)
    return decorator

def balance_checker(allow_negative: bool = False):
    """
    Function decorator factory for balance checking.
    Usage: @balance_checker(allow_negative=False)
    """
    def decorator(func):
        return BalanceCheckDecorator(func, allow_negative)
    return decorator
 
 
def audit_trail(func):
    """
    Function decorator for audit trail.
    Usage: @audit_trail
    """
    decorator = AuditDecorator(func)
    return decorator
 
 
# Composite decorator that combines multiple decorators
def full_transaction_decorator(min_amount: float = 0.01, max_amount: float = 1000000, allow_negative: bool = False):
    """
    Composite decorator that applies validation, balance check, logging, and audit.
    
    Usage: @full_transaction_decorator(min_amount=1, max_amount=5000)
    """
    def decorator(func):
        # Apply decorators in order: validate → check balance → log → audit
        decorated = func
        decorated = AuditDecorator(decorated)
        decorated = LoggingDecorator(decorated)
        decorated = BalanceCheckDecorator(decorated, allow_negative)
        decorated = ValidationDecorator(decorated, min_amount, max_amount)
        return decorated
    return decorator