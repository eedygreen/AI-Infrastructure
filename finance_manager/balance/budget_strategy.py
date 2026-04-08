
from abc import ABC, abstractmethod
from typing import Dict

class BudgetStrategy(ABC):
    """
    Abstract base class for budget planning strategies.
    Implement Strategy Pattern to allow different budgeting approaches
    """

    @abstractmethod
    def allocate(self, income: float) -> Dict[str, float]:
        """
        Allocate income according to the budgeting strategy

        Args:   
            income: Total monthly income
        
        Returns:
            Dictionary mapping category names to allocated amounts
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return a description of this strategy."""
        pass

class FiftyThirtyTwentyStrategy(BudgetStrategy):
    """
    50/30/20 Rule:
    - 50% Needs         (essentials)
    - 30% Wants         (entertainment)
    - 20% Savings/Debt  (emergency)
    """

    def allocate(self, income: float) -> Dict[str, float]:
        """Allocate income using the 50/30/20."""
        return {
            "Needs": income * 0.50,
            "Wants": income * 0.30,
            "Savings": income * 0.20
        }

    def get_strategy_name(self) -> str:
        return "50/30/20 Rule"
    
    def get_description(self) -> str:
        return "50% Needs, 30% Wants, 20% Savings/Debt"
    
class ZeroBasedBudgetStrategy(BudgetStrategy):
    """
    Zero-Based Budgeting:
    Earning(s) is assigned a purpose. Income - Expenses = 0
    
    Typical allocation:
    - 35% Housing (rent/mortgage, utilities, maintenance)
    - 15% Transportation (car payment, gas, insurance)
    - 15% Food (groceries, dining out)
    - 10% Savings (emergency fund, retirement)
    - 10% Insurance (health, life, disability)
    - 5% Entertainment (hobbies, subscriptions)
    - 5% Personal (clothing, personal care)
    - 5% Miscellaneous (gifts, donations, unexpected)
    """

    def allocate(self, income: float) -> Dict[str, float]:
        """Allocate income using zero-base budgeting."""
        return {
            "Housing": income * 0.35,
            "Transportation": income * 0.15,
            "Food": income * 0.15,
            "Savings": income * 0.10,
            "Insurance": income * 0.10,
            "Entertainment": income * 0.05,
            "Personal": income * 0.05,
            "Miscellaneous": income * 0.05
        }
    
    def get_strategy_name(self) -> str:
        return "Zero-Based Budgeting"
    
    def get_description(self) -> str:
        return "Every Earning asssigned a purpose, Income - Expenses = 0"
    
class EnvelopeBudgetingStrategy(BudgetStrategy):
    """
    Envelope Budgeting:
    Cash is divided into envelopes for different spending categories.
    Once an envelope is empty, no more spending in that category.

    Allocation:
    - 30% Groceries
    - 25% Bills (utilities, phone, internet)
    - 15% Transportation (gass, public transit)
    - 10% Enterntainment
    - 10% Savings
    - 10% Miscellaneous
    """

    def allocate(self, income: float) -> Dict[str, float]:
        """Allocate income using envelope budgeting."""
        return {
            "Groceries": income * 0.30,
            "Bills": income * 0.25,
            "Transportation": income * 0.15,
            "Entertainment": income * 0.10,
            "Savings": income * 0.10,
            "Miscellaneous": income * 0.10
        }
    
    def get_strategy_name(self) -> str:
        return "Envelope Budgeting"
    
    def get_description(self) -> str:
        return "Cash divided into category envelopes"
    
class AggressiveSavingsStrategy(BudgetStrategy):
    """
    Aggresive Savings:
    Maximizes savings and investments for financial independence.

    Allocation:
    - 40% Essential Expenses (Housing, Food, Utilities)
    - 40% Savings/Investments (retirement, emergency fund)
    - 15% Debt Repayment (loans, credite cards)
    - 5% Discretionary (minimal spending on wants)
    """

    def allocate(self, income: float) -> Dict[str, float]:
        """ALlocate income with aggressive saings focus."""
        return {
            "Essential Expenses": income * 0.40,
            "Savings/Investments": income * 0.40,
            "Debt Repayment": income * 0.15,
            "Discretionary": income * 0.05
        }
    
    def get_strategy_name(self) -> str:
        return "Aggressive Savings"
    
    def get_description(self) -> str:
        return "Maximize savings (40%) for financial independence"
    
class BudgetPlanner:
    """
    Context Class that uses a BudgetStrategy to plan Budgets.
    Allows switching between different budgeting strategies at runtime.
    """

    def __init__(self, strategy: BudgetStrategy = None):
        """
        Initialize with a budgeting strategy.

        Args: 
            strategy: The budgeting strategy to use (default: 50/30/20)
        """
        self._strategy = strategy or FiftyThirtyTwentyStrategy()

    def set_strategy(self, strategy: BudgetStrategy):
        """
        Change the budgeting strategy

        Args:
            strategy: New budgeting strategy to use
        """
        self._strategy = strategy

    def get_strategy(self) -> BudgetStrategy:
        """Get the current budgeting strategy"""
        return self._strategy
    
    def create_budget(self, income: float) -> Dict[str, float]:
        """
        Create a budget allocation based on the current strategy.

        Args:
            income: Monthly income amount

        Returns:    
            Dictionary mapping categories to allocated amounts
        """
        if income <= 0:
            raise ValueError("Income must be greater than zero")
        
        return self._strategy.allocate(income)
    
    def get_budget_summary(self, income: float) -> str:
        """
        Get a formatted summary of the budget allocation.

        Args:
            income: Monthly income amount

        Returns: 
            Formatted string with budget breakdown
        """

        budget = self.create_budget(income)

        summaries = [
            f"Budget Strategy: {self._strategy.get_strategy_name()}",
            f"Description: {self._strategy.get_description()}",
            f"Monthly Income: ${income:.2f}",
            "-" * 50
        ]

        for category, amount in budget.items():
            percentage = (amount / income) * 100
            summaries.append(f"{category:20s} ${amount:10.2f} {percentage:5.1f}%")
        
        summaries.append("-" * 50)
        total = sum(budget.values())
        summaries.append(f"{'Total Allocate':20s} ${total:10.2f} (100.0%)")

        return "\n".join(summaries)