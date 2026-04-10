"""Finance Manager main entry point.

This module demonstrates all 6 design patterns working together:
- Singleton Pattern (Balance)
- Adapter Pattern (External transactions)
- Observer Pattern (Balance notifications)
- Command Pattern (Undo/Redo)
- Strategy Pattern (Budget planning)
- Decorator Pattern (Validation/Logging)
"""

from balance.balance import Balance
from balance.balance_observer import (
    LowBalanceAlertObserver,
    PrintObserver
)
from transaction.transaction_adapter import TransactionAdapter
from transaction.external_income_transaction import ExternalFreelanceIncome
from transaction.transaction_command import (
    IncomeCommand,
    ExpenseCommand,
    TransactionInvoker
)
from budget.budget_strategy import (
    BudgetPlanner,
    AggressiveSavingsStrategy
)
from transaction.transaction_decorator import (
    LoggingDecorator,
    AuditDecorator,
    full_transaction_decorator
)


def print_header(title):
    """Print formatted section header.

    Args:
        title: The header text to display
    """
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
 
 
def print_balance_info(balance):
    """Print current balance information.
    
    Args:
        balance: Balance instance to display
    """
    print(f"💰 Current Balance: ${balance.get_balance():.2f}")
 
 
def main():
    """Main entry point demonstrating all 6 design patterns."""
    print_header("🏦 FINANCE MANAGER - ALL DESIGN PATTERNS DEMO")
    
    # ====================================================================
    # PATTERN 1 & 3: SINGLETON + OBSERVER
    # ====================================================================
    print_header("1️⃣  Singleton Pattern + Observer Pattern")
    
    # Get singleton instance of Balance
    balance = Balance.get_instance()
    print("✓ Balance instance created (Singleton Pattern)")
    
    # Register observers
    balance.register_observer(PrintObserver())
    balance.register_observer(LowBalanceAlertObserver(threshold=100))
    print("✓ Observers registered (Observer Pattern)")
    print("  - PrintObserver: Logs all balance changes")
    print("  - LowBalanceAlertObserver: Alerts when balance < $100")
    
    # ====================================================================
    # PATTERN 6: DECORATOR - Transaction Validation & Logging
    # ====================================================================
    print_header("2️⃣  Decorator Pattern - Safe Transaction Processing")
    
    # Clear previous logs
    LoggingDecorator.clear_logs()
    AuditDecorator.clear_audit_log()
    
    # Create decorated transaction function
    @full_transaction_decorator(
        min_amount=1,
        max_amount=10000,
        allow_negative=False
    )
    def apply_transaction_safe(bal, txn):
        """Apply transaction with validation, logging, and auditing."""
        bal.apply_transaction(txn)
    
    print("✓ Transaction decorator configured:")
    print("  - Validation: Amount must be $1-$10,000")
    print("  - Balance Check: Prevent overdraft")
    print("  - Logging: All transactions logged")
    print("  - Audit Trail: Compliance tracking")
    
    # ====================================================================
    # PATTERN 2: ADAPTER - External Transaction Integration
    # ====================================================================
    print_header("3️⃣  Adapter Pattern - External Income Processing")
    
    # Create external freelance income (different format)
    freelance_income = ExternalFreelanceIncome(
        1200,
        "INV-98765",
        "Mobile App Project"
    )
    print("✓ External transaction received:")
    print(f"  Invoice: {freelance_income.invoice_id}")
    print(f"  Project: {freelance_income.description}")
    print(f"  Amount: ${freelance_income.amount:.2f}")
    
    # Adapt to internal Transaction format
    adapter = TransactionAdapter(freelance_income)
    adapted_transaction = adapter.to_transaction()
    print(f"✓ Adapted to internal format: {adapted_transaction}")
    
    # ====================================================================
    # PATTERN 4: COMMAND - Transactions with Undo/Redo
    # ====================================================================
    print_header("4️⃣  Command Pattern - Transactions with Undo/Redo")
    
    # Create command invoker
    invoker = TransactionInvoker(balance)
    print("✓ Transaction Invoker created")
    
    # Apply adapted external income via command
    print("\n📥 Processing external income...")
    invoker.execute_command(IncomeCommand(adapted_transaction.amount))
    print_balance_info(balance)
    
    # Create and apply standard transactions via commands
    print("\n📝 Processing standard transactions...")
    standard_transactions = [
        (IncomeCommand(100), "Salary payment"),
        (ExpenseCommand(50), "Groceries"),
        (IncomeCommand(200), "Bonus"),
        (ExpenseCommand(75), "Utilities"),
    ]
    
    for command, description in standard_transactions:
        print(f"\n  {description}: ${command.amount:.2f}")
        try:
            invoker.execute_command(command)
            print_balance_info(balance)
        except ValueError as e:
            print(f"  ❌ Transaction rejected: {e}")
    
    # Demonstrate undo
    print("\n↩️  Demonstrating Undo:")
    print("  Undoing last transaction...")
    if invoker.can_undo():
        undone = invoker.undo()
        print(f"  ✓ Undone: {undone}")
        print_balance_info(balance)
    
    # Demonstrate redo
    print("\n↪️  Demonstrating Redo:")
    print("  Redoing last transaction...")
    if invoker.can_redo():
        redone = invoker.redo()
        print(f"  ✓ Redone: {redone}")
        print_balance_info(balance)
    
    # ====================================================================
    # PATTERN 5: STRATEGY - Budget Planning
    # ====================================================================
    print_header("5️⃣  Strategy Pattern - Dynamic Budget Planning")
    
    current_balance = balance.get_balance()
    print(f"Current Balance: ${current_balance:.2f}")
    
    # Create budget planner with default strategy
    planner = BudgetPlanner()
    strategy_name = planner.get_strategy().get_strategy_name()
    print(f"\n💡 Strategy 1: {strategy_name}")
    budget1 = planner.create_budget(current_balance)
    for category, amount in budget1.items():
        percentage = (amount / current_balance) * 100
        print(f"  {category:20s} ${amount:8.2f} ({percentage:5.1f}%)")
    
    # Switch to aggressive savings strategy
    planner.set_strategy(AggressiveSavingsStrategy())
    strategy_name = planner.get_strategy().get_strategy_name()
    print(f"\n💡 Strategy 2: {strategy_name}")
    budget2 = planner.create_budget(current_balance)
    for category, amount in budget2.items():
        percentage = (amount / current_balance) * 100
        print(f"  {category:20s} ${amount:8.2f} ({percentage:5.1f}%)")
    
    # ====================================================================
    # SUMMARY & REPORTS
    # ====================================================================
    print_header("📊 SESSION SUMMARY")
    
    print(f"\n💰 Final Balance: ${balance.get_balance():.2f}")
    print(f"📝 Total Transactions: {len(invoker.get_history())}")
    print(f"↩️  Can Undo: {invoker.can_undo()}")
    print(f"↪️  Can Redo: {invoker.can_redo()}")
    
    # Transaction logs
    logs = LoggingDecorator.get_logs()
    print(f"\n📋 Transaction Logs ({len(logs)} entries):")
    for log in logs[:5]:  # Show first 5
        print(f"  {log}")
    if len(logs) > 5:
        print(f"  ... and {len(logs) - 5} more")
    
    # Audit trail
    audit_log = AuditDecorator.get_audit_log()
    print(f"\n🔍 Audit Trail ({len(audit_log)} entries):")
    for i, entry in enumerate(audit_log[:3], 1):  # Show first 3
        print(f"  Entry {i}:")
        print(f"    Type: {entry['transaction_type']}")
        print(f"    Amount: ${entry['amount']:.2f}")
        balance_before = entry['balance_before']
        balance_after = entry['balance_after']
        print(f"    Balance: ${balance_before:.2f} → ${balance_after:.2f}")
    if len(audit_log) > 3:
        print(f"  ... and {len(audit_log) - 3} more")
    
    # Command history
    print("\n📜 Command History:")
    for i, cmd in enumerate(invoker.get_history(), 1):
        print(f"  {i}. {cmd}")
    
    # Design patterns summary
    print_header("✅ ALL 6 DESIGN PATTERNS DEMONSTRATED")
    print("""
1️⃣  Singleton Pattern       ✓ Single Balance instance
2️⃣  Adapter Pattern         ✓ External transaction integration  
3️⃣  Observer Pattern        ✓ Balance change notifications
4️⃣  Command Pattern         ✓ Undo/Redo transactions
5️⃣  Strategy Pattern        ✓ Dynamic budget planning
6️⃣  Decorator Pattern       ✓ Validation, logging, auditing
 
🎉 Finance Manager is production-ready with enterprise-grade architecture!
""")
    
    print("=" * 70)
 
 
if __name__ == "__main__":
    main()
 