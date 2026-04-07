# This module contains the OrderTracker class, which encapsulates the core
# business logic for managing orders.

class OrderTracker:
    """
    Manages customer orders, providing functionalities to add, update,
    and retrieve order information.
    """
    VALID_STATUSES = {"pending", "processing", "shipped", "delivered", "cancelled"}

    def __init__(self, storage):
        required_methods = ['save_order', 'get_order', 'get_all_orders', 'delete_order']
        for method in required_methods:
            if not hasattr(storage, method) or not callable(getattr(storage, method)):
                raise TypeError(f"Storage object must implement a callable '{method}' method.")
        self.storage = storage

    def _validate_status(self, status: str):
        """Validate the status is one of the allowed values"""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Status must be one of {self.VALID_STATUSES}, got '{status}'")
    
    def add_order(self, order_id: str, item_name: str, quantity: int, customer_id: str, status: str = "pending"):
        """
        Adds a new order to the system

        Args:
            order_id: Unique Identifier for the order
            item_name: Name of the item being ordered
            quantity: Number of items (must be positive)
            customer_id: Unique identifier for the customer
            status: Order status (default: "pending")
        Raises:
            ValueError: If any validation fails
        """

        if not order_id or not order_id.strip():
            raise ValueError("Order ID cannot be empty.")
        if not item_name or not item_name.strip():
            raise ValueError("Item name cannot be empty.")
        if not customer_id or not customer_id.strip():
            raise ValueError("Customer ID cannot be empty")
        
        if quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        
        self._validate_status(status)

        existing_order = self.storage.get_order(order_id)
        if existing_order is not None:
            raise ValueError(f"Order with ID '{order_id}' already exists.")
        
        order = {
            "order_id": order_id,
            "item_name": item_name,
            "quantity": quantity,
            "customer_id": customer_id,
            "status": status
        }
        self.storage.save_order(order_id, order)

    def get_order_by_id(self, order_id: str):
        """
        Retreives an order by its ID.

        Args:
            order_id: The ID of the order to retrieve

        Returns:
            The order dict if found, None otherwise

        Raises:
            ValueError: if order_id is empty
        """
        if not order_id or not order_id.strip():
            raise ValueError("Order ID cannot be empty.")
        
        result = self.storage.get_order(order_id)
        
        return result
    
    def update_order_status(self, order_id: str, new_status: str):
        """
        Updates the status of an existing order.

        Args:
            order_id: The ID of the order to update
            new_status: The new status to set

        Raises:
            ValueError: If validation failes or order not found
        """

        if not order_id or not order_id.strip():
            raise ValueError("Order ID cannot be empty.")
        
        # validate status - fail fast before checking storage
        self._validate_status(new_status)

        order = self.storage.get_order(order_id)

        if order is None:
            raise ValueError(f"Order with ID '{order_id}' not found")
        
        order["status"] = new_status
        self.storage.save_order(order_id, order)


    def list_all_orders(self):
        """
        Returns all orders as a list.

        Returns:
            List of all order dicts
        """
        orders_dict = self.storage.get_all_orders()
        return list(orders_dict.values())

    def list_orders_by_status(self, status: str):
        """
        Return all orders with th specified status

        Args:
            status: The status to filter by

        Returns:
            List of order dicts matching the status

        Raises:
            valueError: If status is empty or invalid
        """
        if not status or not status.strip():
            raise ValueError("Status cannot be empty.")
        
        self._validate_status(status)

        order_dict = self.storage.get_all_orders()

        return [order for order in order_dict.values() if order.get("status") == status]

    def delete_order(self, order_id: str):
        """
        Deletes an order from the system.
        
        Args:
            order_id: The ID of the order to delete
            
        Returns:
            The deleted order dict
            
        Raises:
            ValueError: If order_id is empty or order doesn't exist
        """

        if not order_id or not order_id.strip():
            raise ValueError("Order ID cannot be empty.")
        
        order = self.storage.get_order(order_id)
        if order is None:
            raise ValueError(f"Order with ID '{order_id}' not found.")
        
        self.storage.delete_order(order_id)

        return order