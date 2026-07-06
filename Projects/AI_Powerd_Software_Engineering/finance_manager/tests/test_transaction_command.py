import unittest
from balance.balance import Balance
from transaction.transaction_command import IncomeCommand, ExpenseCommand, TransactionInvoker

class TestTransactionCommand(unittest.TestCase):
    def setUp(self):
        """Reset balance signleton before each test."""
        Balance._instance = None
        self.balance = Balance.get_instance()
        self.balance.reset()
    
    def test_income_command_execution(self):
        cmd = IncomeCommand(100)
        cmd.execute(self.balance)

        self.assertEqual(self.balance.get_balance(), 100)

    def test_income_command_undo(self):
        self.balance.add_income(100)

        cmd = IncomeCommand(50)
        cmd.undo(self.balance)

        self.assertEqual(self.balance.get_balance(), 50)

    def test_expense_command_execute(self):
        self.balance.add_income(100)

        cmd = ExpenseCommand(70)
        cmd.execute(self.balance)

        self.assertEqual(self.balance.get_balance(), 30)

    def test_expense_command_undo(self):
        self.balance.add_income(200)

        cmd = ExpenseCommand(120)
        cmd.execute(self.balance)
        self.assertEqual(self.balance.get_balance(), 80)

        cmd.undo(self.balance)
        self.assertEqual(self.balance.get_balance(), 200)

    def test_undo_redo_stack_management(self):
        invoker = TransactionInvoker(self.balance)

        invoker.execute_command(IncomeCommand(100))
        self.assertEqual(invoker.can_undo(), True)
        self.assertEqual(invoker.can_redo(), False) 

        invoker.undo()
        self.assertEqual(invoker.can_undo(), False)
        self.assertEqual(invoker.can_redo(), True)

    def test_multiple_commands_with_invoker(self):
        invoker = TransactionInvoker(self.balance)

        invoker.execute_command(IncomeCommand(100))
        invoker.execute_command(ExpenseCommand(30))
        invoker.execute_command(IncomeCommand(50))

        self.assertEqual(self.balance.get_balance(), 120)
        self.assertEqual(len(invoker.get_history()), 3)

    def test_undo_multiple_commands(self):
        invoker = TransactionInvoker(self.balance)

        invoker.execute_command(IncomeCommand(100))
        invoker.execute_command(ExpenseCommand(30))
        invoker.execute_command(IncomeCommand(50))

        self.assertEqual(self.balance.get_balance(), 120)

        invoker.undo()     # undo the last IncomeCommand(50)
        self.assertEqual(self.balance.get_balance(), 70)

        invoker.undo()     # undo the second command ExpenseCommand(30)
        self.assertEqual(self.balance.get_balance(), 100)

        invoker.undo()     # undo the first command IncomeCommand(100)
        self.assertEqual(self.balance.get_balance(), 0)

    def test_redo_after_undo(self):
        invoker = TransactionInvoker(self.balance)

        invoker.execute_command(IncomeCommand(100))
        invoker.execute_command(ExpenseCommand(30))

        self.assertEqual(self.balance.get_balance(), 70)

        invoker.undo()     # undo both
        invoker.undo()
        self.assertEqual(self.balance.get_balance(), 0)

        invoker.redo()     # redo both
        self.assertEqual(self.balance.get_balance(), 100)

        invoker.redo()
        self.assertEqual(self.balance.get_balance(), 70)

    def test_new_command_clears_redo_stack(self):
        invoker = TransactionInvoker(self.balance)

        invoker.execute_command(IncomeCommand(100))
        invoker.execute_command(ExpenseCommand(30))

        invoker.undo()
        self.assertTrue(invoker.can_redo())

        invoker.execute_command(IncomeCommand(50))     # Execute new command - should clear reod stack
        self.assertFalse(invoker.can_redo())
        self.assertEqual(self.balance.get_balance(), 150)

    def test_undo_with_no_history_raises_error(self):
        invoker = TransactionInvoker(self.balance)

        with self.assertRaises(ValueError) as context:
            invoker.undo()

        self.assertEqual(str(context.exception), "No commands to undo")

    def test_redo_with_no_history_raises_error(self):
        invoker = TransactionInvoker(self.balance)

        with self.assertRaises(ValueError) as context:
            invoker.redo()

        self.assertEqual(str(context.exception), "No command to redo")
    
    def test_command_history_tracking(self):
        invoker = TransactionInvoker(self.balance)

        cmd1 = IncomeCommand(100)
        cmd2 = ExpenseCommand(30)
        cmd3 = IncomeCommand(50) 

        invoker.execute_command(cmd1)
        invoker.execute_command(cmd2)
        invoker.execute_command(cmd3)

        history = invoker.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0], cmd1)
        self.assertEqual(history[1], cmd2)
        self.assertEqual(history[2], cmd3)

    def test_clear_history(self):
        invoker = TransactionInvoker(self.balance)

        invoker.execute_command(IncomeCommand(100))
        invoker.execute_command(ExpenseCommand(30))
        invoker.undo()

        self.assertTrue(invoker.can_undo())
        self.assertTrue(invoker.can_redo())

        invoker.clear_history()

        self.assertFalse(invoker.can_undo())
        self.assertFalse(invoker.can_redo())
        self.assertEqual(len(invoker.get_history()), 0)

    def test_get_transaction(self):
        income_cmd = IncomeCommand(100)
        expense_cmd = ExpenseCommand(50)

        income_txn = income_cmd.get_transaction()
        expense_txn = expense_cmd.get_transaction()

        self.assertEqual(income_txn.amount, 100)
        self.assertEqual(expense_txn.amount, 50)

        from transaction.transaction_category import TransactionCategory
        self.assertEqual(income_txn.category, TransactionCategory.INCOME)
        self.assertEqual(expense_txn.category, TransactionCategory.EXPENSE)
    

if __name__ == "__main__":
    unittest.main()