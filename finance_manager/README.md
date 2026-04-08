# Finance Manager README

## Overview

Complete Architectural Design for a Finance Manager application demonstrating 6 design patterns:
1. Singleton Pattern (Balance)
2. Adapter Pattern (External Transactions)
3. Observer Pattern (Notifications)
4. Command Pattern (Undo/Redo)
5. Strategy Pattern (Budget Planning)
6. Decorator Pattern (Validation/Logging) is organized into separate modules for clarity and maintainability. Comprehensive test suite included for all patterns.

## Complete Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
           ┌─────────────────┴────────────────┐
           │                                   │
           ▼                                   ▼
    ┌─────────────┐                   ┌──────────────┐
    │  Budget     │                   │ Transaction  │
    │  Planner    │◄───Strategy       │ Invoker      │◄───Command
    └─────────────┘    Pattern        └──────┬───────┘    Pattern
           │                                  │
           │                           ┌──────┴──────────┐
           │                           │                  │
           ▼                           ▼                  ▼
    ┌─────────────┐           ┌──────────────┐  ┌──────────────┐
    │  50/30/20   │           │   Income     │  │   Expense    │
    │  Zero-Based │           │   Command    │  │   Command    │
    │  Envelope   │           └──────────────┘  └──────────────┘
    │  Aggressive │                  │                  │
    └─────────────┘                  └────────┬─────────┘
                                               │
                                               ▼
                              ┌─────────────────────────────────┐
                              │  Transaction Decorators         │
                              │  • Validation                   │
                              │  • Balance Check                │
                              │  • Logging                      │
                              │  • Audit                        │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │   Transaction Adapter           │◄───Adapter
                              │   (External Systems)            │    Pattern
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │   Balance (Singleton)           │◄───Singleton
                              │   • apply_transaction()         │    Pattern
                              │   • notify_observers()          │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │   Observers                     │◄───Observer
                              │   • PrintObserver               │    Pattern
                              │   • LowBalanceAlertObserver     │
                              └─────────────────────────────────┘
```
## Directory Structure:
```
finance_manager/
├── main.py                          # Entry point (runs the demo)
├── balance/
│   ├── __init__.py
│   ├── balance.py                   # Singleton Pattern
│   └── balance_observer.py          # Observer Pattern
├── transaction/
│   ├── __init__.py
│   ├── transaction.py               # Core Transaction class
│   ├── transaction_category.py      # INCOME/EXPENSE enum
│   ├── transaction_adapter.py       # Adapter Pattern
│   ├── transaction_command.py       # Command Pattern (undo/redo)
│   ├── transaction_decorator.py     # Decorator Pattern (validation/logging)
│   └── external_income_transaction.py  # External format
├── budget/
│   ├── __init__.py
│   └── budget_strategy.py           # Strategy Pattern
└── tests/
    ├── test_balance.py
    ├── test_budget_strategy.py
    ├── test_transaction_command.py
    ├── test_transaction_decorator.py
    ├── test_transaction_adapter.py
    ├──test_transaction.py
    └── test_balance_observer.py
```

---

## Files Included

### **Implementation Files:**
1. `balance.py` - Singleton pattern with observer support
2. `balance_observer.py` - Observer implementations
3. `transaction.py` - Core Transaction class
4. `transaction_category.py` - INCOME/EXPENSE enum
5. `transaction_adapter.py` - Adapter pattern for external formats
6. `transaction_command.py` - Command pattern with undo/redo
7. `transaction_decorator.py` - Decorator pattern for validation/logging
8. `budget_strategy.py` - Strategy pattern for budget planning
9. `external_income_transaction.py` - External transaction format
10. `main.py` - Entry point demonstrating all patterns

### **Test Files:**
1. `test_balance.py` - 8 tests for Singleton & Observer
2. `test_budget_strategy.py` - 12 tests for Strategy pattern
3. `test_transaction_command.py` - 14 tests for Command pattern
4. `test_transaction_decorator.py` - 13 tests for Decorator pattern

### **Documentation Files:**
1. `README.md` - Project overview and setup instructions
2. `PROJECT_SUMMARY.md` - Complete project overview
3. `DESIGN_PATTERNS.md` - Detailed explanation of each design pattern used
4. `COMMAND_PATTERN.md` - In-depth look at Command pattern implementation
5. `STRATEGY_DECORATOR_PATTERN.md` - In-depth look at Strategy and Decorator  pattern implementation



What main.py Does
The main.py file demonstrates all 6 design patterns working together:

### **1. Singleton Pattern** (Balance)
- Creates single instance of Balance
- Ensures data consistency

### **2. Adapter Pattern** (External Transactions)
- Converts `ExternalFreelanceIncome` to `Transaction`
- Integrates external systems

### **3. Observer Pattern** (Notifications)
- Registers `PrintObserver` and `LowBalanceAlertObserver`
- Gets notified on all balance changes

### **4. Command Pattern** (Undo/Redo)
- Executes transactions as commands
- Demonstrates undo and redo functionality
- Maintains command history

### **5. Strategy Pattern** (Budget Planning)
- Creates budgets using different strategies
- Switches between 50/30/20 and Aggressive Savings
- Shows dynamic strategy selection

### **6. Decorator Pattern** (Validation/Logging)
- Validates all transactions (amount limits, overdraft protection)
- Logs every transaction with timestamps
- Creates audit trail for compliance

---

## Step 1: clone the repository and navigate to the finance_manager folder

```bash
git clone <repository-url>
```

## Step 2: Running Tests

```bash
cd finance_manager
```

## Step 3: Run all tests
```bash
python -m pytest tests/ -v

# Or run individual test files
python tests/test_balance.py
python tests/test_budget_strategy.py
python tests/test_transaction_command.py
python tests/test_transaction_decorator.py
```

---
### Step 4: Run the Application
```bash
python main.py
```

---

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, ensure:
1. You're in the correct directory (`finance_manager/`)
2. All `__init__.py` files exist
3. Directory structure matches the layout above

### Test Failures

If tests fail:
1. Run `python -m pytest tests/ -v` to see detailed output
2. Check that all dependencies are installed
3. Verify file locations match expected structure

---

## Support

For questions or issues:
- Review the comprehensive documentation in the `.md` files
- Check test files for usage examples
- Examine `main.py` for integration examples

🎉 **Enjoy your fully-functional Finance Manager with enterprise-grade architecture!**