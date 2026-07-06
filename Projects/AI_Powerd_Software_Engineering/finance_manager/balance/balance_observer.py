# balance_observer.py

class IBalanceObserver:
    def update(self, balance, transaction):
        """Handle balance updates."""
        raise NotImplementedError("Subclasses must implement update method.")

class PrintObserver(IBalanceObserver):
    def update(self, balance, transaction):
        """Print balance update message."""
        current_balance = balance.get_balance()
        print(f"Balance updated: {transaction} | New Balance: ${current_balance:.2f}")

class LowBalanceAlertObserver(IBalanceObserver):
    def __init__(self, threshold):
        self.threshold = threshold
        self.alert_triggered = False
        self._was_below_threshold = False

    def update(self, balance, transaction):
        """Alert if balance drops below threshold."""
        current_balance = balance.get_balance()
        is_below_threshold  = current_balance < self.threshold

        # Alert triggers only on transaction from above to below threshold
        if not self._was_below_threshold and is_below_threshold:
            self.alert_triggered = True

        # Reset alert when balance goes back above threshold
        elif self._was_below_threshold and not is_below_threshold:
            self.alert_triggered = False

        # update state for next check
        self._was_below_threshold = is_below_threshold


    
