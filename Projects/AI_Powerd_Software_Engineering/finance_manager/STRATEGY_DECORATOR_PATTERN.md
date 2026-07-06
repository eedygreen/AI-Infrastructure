# Strategy and Decorator Patterns - Complete Documentation

## Overview

This document covers the implementation of **Strategy Pattern** for budget planning and **Decorator Pattern** for transaction validation and logging in the Finance Manager application.

---

## Design Pattern #5: Strategy Pattern 📊

### **Pattern Chosen:** Strategy Pattern for Budget Planning

### **Why This Pattern?**

The Strategy Pattern was chosen for budget planning because:

1. **Different budgeting philosophies**: People budget differently based on their financial goals
   - Some prioritize savings (Aggressive Savings)
   - Some prefer balanced approach (50/30/20 Rule)
   - Some need detailed categories (Zero-Based Budgeting)

2. **Runtime flexibility**: Users should be able to switch strategies without restarting
3. **Easy extensibility**: New budgeting methods can be added without modifying existing code
4. **Encapsulation**: Each strategy contains its own allocation logic

### **Where It Fits in the Application**

```
User Input (Income)
       ↓
BudgetPlanner (Context)
       ↓
┌──────────────────────────────────┐
│   BudgetStrategy (Interface)     │
└──────────────────────────────────┘
       ↓
┌──────────────┬─────────────────┬──────────────────┬─────────────────┐
│  50/30/20    │  Zero-Based     │   Envelope       │   Aggressive    │
│   Strategy   │   Strategy      │   Strategy       │   Savings       │
└──────────────┴─────────────────┴──────────────────┴─────────────────┘
       ↓
Budget Allocation (Dictionary)
```

### **Implementation**

```python
from abc import ABC, abstractmethod

class BudgetStrategy(ABC):
    """Abstract base for budget strategies."""
    
    @abstractmethod
    def allocate(self, income: float) -> Dict[str, float]:
        """Allocate income to categories."""
        pass

class FiftyThirtyTwentyStrategy(BudgetStrategy):
    """50% Needs, 30% Wants, 20% Savings."""
    
    def allocate(self, income: float) -> Dict[str, float]:
        return {
            "Needs": income * 0.50,
            "Wants": income * 0.30,
            "Savings": income * 0.20
        }

class BudgetPlanner:
    """Context that uses a strategy."""
    
    def __init__(self, strategy: BudgetStrategy = None):
        self._strategy = strategy or FiftyThirtyTwentyStrategy()
    
    def set_strategy(self, strategy: BudgetStrategy):
        self._strategy = strategy
    
    def create_budget(self, income: float) -> Dict[str, float]:
        return self._strategy.allocate(income)
```

### **How It Improves the Application**

#### **1. Flexibility** 🎯

Users can switch between budgeting strategies at runtime:

```python
planner = BudgetPlanner()

# Start with 50/30/20
budget1 = planner.create_budget(5000)

# Switch to aggressive savings
planner.set_strategy(AggressiveSavingsStrategy())
budget2 = planner.create_budget(5000)

# Switch to zero-based
planner.set_strategy(ZeroBasedBudgetingStrategy())
budget3 = planner.create_budget(5000)
```

**Benefits:**
- No restart required
- Try different strategies instantly
- Compare allocations side-by-side

#### **2. Testability** ✅

Each strategy is independently testable:

```python
def test_fifty_thirty_twenty_allocation():
    strategy = FiftyThirtyTwentyStrategy()
    budget = strategy.allocate(3000)
    
    assert budget["Needs"] == 1500
    assert budget["Wants"] == 900
    assert budget["Savings"] == 600
```

**Benefits:**
- Isolated test cases
- Easy to verify calculations
- Mock strategies for integration tests

#### **3. Scalability** 📈

Adding new strategies is trivial:

```python
class StudentBudgetStrategy(BudgetStrategy):
    """Budget for students: minimize expenses, maximize savings."""
    
    def allocate(self, income: float) -> Dict[str, float]:
        return {
            "Essentials": income * 0.60,
            "Books/Education": income * 0.20,
            "Savings": income * 0.15,
            "Fun": income * 0.05
        }

# Use immediately
planner.set_strategy(StudentBudgetStrategy())
```

**Benefits:**
- No modification to existing code
- Open/Closed Principle
- Grows with user needs

---

## Design Pattern #6: Decorator Pattern 🛡️

### **Pattern Chosen:** Decorator Pattern for Transaction Validation & Logging

### **Why This Pattern?**

The Decorator Pattern was chosen for transaction operations because:

1. **Layered functionality**: Validation, logging, and auditing are separate concerns
2. **Composability**: Stack decorators for combined behavior
3. **Non-invasive**: Add features without modifying core transaction logic
4. **Separation of concerns**: Each decorator has one job

### **Where It Fits in the Application**

```
apply_transaction()
       ↓
ValidationDecorator
       ↓
BalanceCheckDecorator
       ↓
LoggingDecorator
       ↓
AuditDecorator
       ↓
Core Transaction Logic
```

### **Implementation**

```python
from abc import ABC, abstractmethod

class TransactionDecorator(ABC):
    """Base decorator for transactions."""
    
    def __init__(self, wrapped_function: Callable):
        self.wrapped_function = wrapped_function
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class ValidationDecorator(TransactionDecorator):
    """Validates transaction amounts."""
    
    def __call__(self, *args, **kwargs):
        transaction = args[1]
        
        if transaction.amount <= 0:
            raise ValueError("Amount must be positive")
        
        if transaction.amount > self.max_amount:
            raise ValueError("Amount exceeds maximum")
        
        return self.wrapped_function(*args, **kwargs)

# Usage
@transaction_validator(min_amount=1, max_amount=10000)
@balance_checker(allow_negative=False)
@transaction_logger
def apply_transaction(balance, transaction):
    balance.apply_transaction(transaction)
```

### **How It Improves the Application**

#### **1. Flexibility** 🎯

Mix and match decorators as needed:

```python
# Minimal: just logging
@transaction_logger
def apply_simple(balance, txn):
    balance.apply_transaction(txn)

# Full protection: validation + balance check + logging + audit
@full_transaction_decorator(min_amount=1, max_amount=5000)
def apply_safe(balance, txn):
    balance.apply_transaction(txn)
```

**Benefits:**
- Choose appropriate level of safety
- Different decorators for different contexts
- Easy to add/remove features

#### **2. Testability** ✅

Each decorator is independently testable:

```python
def test_validation_rejects_negative():
    @transaction_validator()
    def apply(balance, txn):
        balance.apply_transaction(txn)
    
    with pytest.raises(ValueError):
        apply(balance, Transaction(-100, INCOME))

def test_balance_check_prevents_overdraft():
    @balance_checker(allow_negative=False)
    def apply(balance, txn):
        balance.apply_transaction(txn)
    
    balance._balance = 50
    with pytest.raises(ValueError):
        apply(balance, Transaction(100, EXPENSE))
```

**Benefits:**
- Test each concern separately
- Clear error messages
- Predictable behavior

#### **3. Scalability** 📈

New decorators integrate seamlessly:

```python
class FraudDetectionDecorator(TransactionDecorator):
    """Detect suspicious transaction patterns."""
    
    def __call__(self, *args, **kwargs):
        transaction = args[1]
        
        if self.is_suspicious(transaction):
            raise ValueError("Transaction flagged for review")
        
        return self.wrapped_function(*args, **kwargs)

class RateLimitDecorator(TransactionDecorator):
    """Limit transactions per time period."""
    
    def __call__(self, *args, **kwargs):
        if self.exceeded_rate_limit():
            raise ValueError("Too many transactions")
        
        return self.wrapped_function(*args, **kwargs)
```

**Benefits:**
- Extend without modifying core
- Stack as many decorators as needed
- Each decorator is reusable

---

## Implemented Strategies

### **1. 50/30/20 Rule**
- 50% Needs (rent, groceries, utilities)
- 30% Wants (entertainment, dining, hobbies)
- 20% Savings (emergency fund, investments)
- **Best for:** Balanced lifestyle

### **2. Zero-Based Budgeting**
- Every dollar assigned a purpose
- 35% Housing, 15% Transportation, 15% Food, 10% Savings, etc.
- **Best for:** Detailed planners

### **3. Envelope Budgeting**
- Cash divided into category envelopes
- 30% Groceries, 25% Bills, 15% Transportation, etc.
- **Best for:** Visual budgeters

### **4. Aggressive Savings**
- 40% Essential Expenses
- 40% Savings/Investments
- 15% Debt Repayment
- 5% Discretionary
- **Best for:** FIRE movement, early retirement

---

## Implemented Decorators

### **1. ValidationDecorator**
- Validates amount > 0
- Enforces min/max limits
- Checks required attributes

### **2. BalanceCheckDecorator**
- Prevents overdraft
- Optional allow_negative flag
- Only checks expenses

### **3. LoggingDecorator**
- Records timestamp
- Logs transaction details
- Tracks balance changes
- Retrievable log history

### **4. AuditDecorator**
- Creates detailed audit trail
- Records before/after balance
- Timestamps in ISO format
- Compliance-ready format

### **5. Composite Decorator**
- Combines all decorators
- Configurable parameters
- Single decorator for full protection

---

## Testing

### **Strategy Pattern Tests: 12/12 Passing** ✅

- Allocation accuracy for each strategy
- Strategy switching at runtime
- Budget summary formatting
- Invalid income handling
- Multiple strategies comparison

### **Decorator Pattern Tests: 13/13 Passing** ✅

- Validation of amounts
- Overdraft prevention
- Logging functionality
- Audit trail creation
- Decorator composition

**Total: 25 comprehensive tests**

---

## Usage Examples

### **Complete Workflow**

```python
from budget_strategy import BudgetPlanner, AggressiveSavingsStrategy
from transaction_decorator import full_transaction_decorator

# 1. Create budget
planner = BudgetPlanner(AggressiveSavingsStrategy())
budget = planner.create_budget(5000)

# 2. Apply transactions with full protection
@full_transaction_decorator(min_amount=1, max_amount=100000)
def process_transaction(balance, transaction):
    balance.apply_transaction(transaction)

# 3. Receive salary
salary = Transaction(5000, TransactionCategory.INCOME)
process_transaction(balance, salary)

# 4. Pay expenses per budget
for category, amount in budget.items():
    expense = Transaction(amount, TransactionCategory.EXPENSE)
    try:
        process_transaction(balance, expense)
    except ValueError as e:
        print(f"Transaction rejected: {e}")

# 5. Review logs and audit
print(LoggingDecorator.get_logs())
print(AuditDecorator.get_audit_log())
```

---

## Benefits Summary

### **Strategy Pattern**
✅ **Flexibility:** Switch budgeting approaches at runtime  
✅ **Extensibility:** Add new strategies without modifying existing code  
✅ **Encapsulation:** Each strategy contains its own logic  
✅ **User Empowerment:** Choose what works for their situation  

### **Decorator Pattern**
✅ **Safety:** Multiple layers of validation and checking  
✅ **Transparency:** Complete logging and audit trail  
✅ **Composability:** Stack decorators for layered functionality  
✅ **Maintainability:** Separation of concerns, single responsibility  

### **Together**
✅ **Robust System:** Flexible planning + safe execution  
✅ **SOLID Principles:** Open/Closed, Single Responsibility  
✅ **Production Ready:** Validation, logging, auditing  
✅ **User Friendly:** Multiple options, clear feedback  

---

## Design Principles Applied

✅ **Single Responsibility:** Each strategy/decorator has one job  
✅ **Open/Closed:** Open for extension, closed for modification  
✅ **Liskov Substitution:** All strategies/decorators are substitutable  
✅ **Interface Segregation:** Minimal, focused interfaces  
✅ **Dependency Inversion:** Depend on abstractions (BudgetStrategy, TransactionDecorator)

---

## Future Enhancements

**Strategy Pattern:**
- AI-driven budget recommendations
- Historical spending-based strategies
- Goal-oriented strategies (house, car, vacation)
- Multi-account budgeting

**Decorator Pattern:**
- Machine learning fraud detection
- Transaction categorization
- Receipt attachment
- Tax optimization hints

This completes the implementation of all 6 design patterns in the Finance Manager application! 🎉
