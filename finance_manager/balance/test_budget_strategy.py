import unittest

from balance.budget_strategy import (
    FiftyThirtyTwentyStrategy,
    ZeroBasedBudgetStrategy,
    EnvelopeBudgetingStrategy,
    AggressiveSavingsStrategy,
    BudgetPlanner
)

class TestBudgetStrategies(unittest.TestCase):
    def test_fifty_thirty_twenty_allocation(self):
        strategy = FiftyThirtyTwentyStrategy()
        budget = strategy.allocate(3000)

        self.assertEqual(budget["Needs"], 1500)
        self.assertEqual(budget["Wants"], 900)
        self.assertEqual(budget["Savings"], 600)
        self.assertEqual(sum(budget.values()), 3000)

    def test_zero_based_budgeting_allocation(self):
        strategy = ZeroBasedBudgetStrategy()
        budget = strategy.allocate(4000)

        self.assertEqual(budget["Housing"], 1400)
        self.assertEqual(budget["Transportation"], 600)
        self.assertEqual(budget["Food"], 600)
        self.assertEqual(sum(budget.values()), 4000)

    def test_envelop_budgeting_allocation(self):
        strategy = EnvelopeBudgetingStrategy()
        budget = strategy.allocate(2000)

        self.assertEqual(budget["Groceries"], 600)
        self.assertEqual(budget["Bills"], 500)
        self.assertEqual(sum(budget.values()), 2000)

    def test_aggresive_savings_allocation(self):
        strategy = AggressiveSavingsStrategy()
        budget = strategy.allocate(5000)

        self.assertEqual(budget["Essential Expenses"], 2000)
        self.assertEqual(budget["Savings/Investments"], 2000)
        self.assertEqual(budget["Debt Repayment"], 750)
        self.assertEqual(budget["Discretionary"], 250)
        self.assertEqual(sum(budget.values()), 5000)

    def test_strategy_names(self):
        strategies = [
            (FiftyThirtyTwentyStrategy(), "50/30/20 Rule"),
            (ZeroBasedBudgetStrategy(), "Zero-Based Budgeting"),
            (EnvelopeBudgetingStrategy(), "Envelope Budgeting"),
            (AggressiveSavingsStrategy(), "Aggressive Savings")
        ]

        for strategy, expected_name in strategies:
            self.assertEqual(strategy.get_strategy_name(), expected_name)

    def test_strategy_descriptions(self):
        strategies = [
            FiftyThirtyTwentyStrategy(),
            ZeroBasedBudgetStrategy(),
            EnvelopeBudgetingStrategy(),
            AggressiveSavingsStrategy()
        ]

        for strategy in strategies:
            description = strategy.get_description()
            self.assertIsInstance(description, str)
            self.assertTrue(len(description), 0)


class TestBudgetPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = BudgetPlanner()

    def test_defualt_strategy(self):
        
        self.assertIsInstance(self.planner.get_strategy(), FiftyThirtyTwentyStrategy)

    def test_set_strategy(self):
        budget1 = self.planner.create_budget(3000)
        self.assertEqual(budget1["Needs"], 1500)

        self.planner.set_strategy(ZeroBasedBudgetStrategy())
        budget2 = self.planner.create_budget(3000)
        self.assertEqual(budget2["Housing"], 1050)

        self.planner.set_strategy(AggressiveSavingsStrategy())
        budget3 = self.planner.create_budget(3000)
        self.assertEqual(budget3["Savings/Investments"], 1200)

    def test_create_budget_with_custom_strategy(self):
        strategy = EnvelopeBudgetingStrategy()
        planner = BudgetPlanner(strategy)

        budget = planner.create_budget(2000)
        self.assertEqual(budget["Groceries"], 600)

    def test_creat_budget_invalid_income(self):
        with self.assertRaises(ValueError):
            self.planner.create_budget(0)

        with self.assertRaises(ValueError):
            self.planner.create_budget(-100)

    def test_budget_summary(self):
        summary = self.planner.get_budget_summary(3000)

        self.assertIn("50/30/20 Rule", summary)
        self.assertIn("$3000.00", summary)
        self.assertIn("Needs", summary)
        self.assertIn("Wants", summary)
        self.assertIn("Savings", summary)

    def test_multiple_strategies_with_same_income(self):
        income = 5000
        allocations = []
        strategies = [
            FiftyThirtyTwentyStrategy(),
            ZeroBasedBudgetStrategy(),
            AggressiveSavingsStrategy()
        ]

        for strategy in strategies:
            self.planner.set_strategy(strategy)
            budget = self.planner.create_budget(income)
            allocations.append(budget)

        # verify they're different
        self.assertNotEqual(allocations[0], allocations[1])
        self.assertNotEqual(allocations[1], allocations[2])

        # verify all allocate full income
        for allocation in allocations:
            self.assertAlmostEqual(sum(allocation.values()), income, places=2)

if __name__ == "__main__":
    unittest.main()
