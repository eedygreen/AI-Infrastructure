
from abc import ABC, abstractmethod
from transaction.transaction import Transaction
from transaction.transaction_category import TransactionCategory

class TransactionCommand(ABC):
    """
    Abstract base class for transaction commands Interface.
    Implements the Command Pattern to encapsulate transaction operation.
    """
    @abstractmethod
    def execute(self, balance):
        """Execute the command on the given balance"""
        pass

    @abstractmethod
    def undo(self, balance):
        """Undo the command on the given balance."""
        pass

    @abstractmethod
    def get_transaction(self):
        """Get the transaction associated with this command"""
        pass

class IncomeCommand(TransactionCommand):
    """Commnad to apply an income transaction."""

    def __init__(self, amount):
        self.amount = amount
        self.transaction = Transaction(amount, TransactionCategory.INCOME)

    def execute(self, balance):
        """Add income to balance"""
        balance.apply_transaction(self.transaction)

    def undo(self, balance):
        """Undo the income by subtracting it"""
        balance.add_expense(self.amount)

    def get_transaction(self):
        return self.transaction
    
    def __str__(self):
        return f"IncomeCommand(${self.amount})"
    
class ExpenseCommand(TransactionCommand):
    """Command to apply an expense transaction."""

    def __init__(self, amount):
        self.amount = amount
        self.transaction = Transaction(amount, TransactionCategory.EXPENSE)

    def execute(self, balance):
        """Subtract expense from the balance"""
        balance.apply_transaction(self.transaction)

    def undo(self, balance):
        """Undo the expense by adding it back"""
        balance.add_income(self.amount)

    def get_transaction(self):
        """Get the expense transaction."""
        return self.transaction
    
    def __str__(self):
        return f"ExpenseCommand(${self.amount})"
    
class TransactionInvoker:
    """
    Invoker executes commands and maintains history for undo/redo.
    Enables transaction rollback and replay capabilities.
    """

    def __init__(self, balance):
        self.balance = balance
        self.history = []
        self.undo_stack = []

    def execute_command(self, command):
        """Execute a command and aadd it to history."""
        command.execute(self.balance)
        self.history.append(command)
        # clear redo stack when new command is executed
        self.undo_stack.clear()

    def undo(self):
        """Undo the last command."""
        if not self.history:
            raise ValueError("No commands to undo")
        
        command = self.history.pop()
        command.undo(self.balance)
        self.undo_stack.append(command)
        return command
    
    def redo(self):
        """Redo the last undone command."""
        if not self.undo_stack:
            raise ValueError("No command to redo")
        
        command = self.undo_stack.pop()
        command.execute(self.balance)
        self.history.append(command)
        return command
    
    def get_history(self):
        """Get the command history."""
        return self.history.copy()
    
    def can_undo(self):
        """Check if undo is possible."""
        return len(self.history) > 0
    
    def can_redo(self):
        """Check if redo is possible."""
        return len(self.undo_stack) > 0
    
    def clear_history(self):
        """Clear all command history."""
        self.history.clear()
        self.undo_stack.clear()
