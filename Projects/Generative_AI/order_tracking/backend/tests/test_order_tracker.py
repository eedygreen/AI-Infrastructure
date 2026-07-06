import pytest
from unittest.mock import Mock
from ..order_tracker import OrderTracker

# --- Fixtures for Unit Tests ---

@pytest.fixture
def mock_storage():
    """
    Provides a mock storage object for tests.
    This mock will be configured to simulate various storage behaviors.
    """
    mock = Mock()
    # By default, mock get_order to return None (no order found)
    mock.get_order.return_value = None
    # By default, mock get_all_orders to return an empty dict
    mock.get_all_orders.return_value = {}
    return mock

@pytest.fixture
def order_tracker(mock_storage):
    """
    Provides an OrderTracker instance initialized with the mock_storage.
    """
    return OrderTracker(mock_storage)

#
# --- TODO: add test functions below this line ---
# === add_order tests ===
def test_add_order_successfully(order_tracker, mock_storage):
    """Tests adding a new order with default 'pending' status. """
    order_tracker.add_order("ORD001", "Laptop", 1, "CUST001")

    mock_storage.save_order.assert_called_once()
    call_args = mock_storage.save_order.call_args[0]

    assert call_args[1]["status"] == "pending"

def test_add_order_with_explicit_status(order_tracker, mock_storage):
    """Tests adding a new order with an explicit status. """
    order_tracker.add_order("ORD002", "Mouse", 2, "CUST002", status="shipped")

    mock_storage.save_order.assert_called_once()
    call_args = mock_storage.save_order.call_args[0]
    assert call_args[1]["status"] == "shipped"

def test_add_order_raises_error_if_exists(order_tracker, mock_storage):
    """Tests that adding an order with a duplicate ID raises a ValueError."""
    # Simulate that the storage finds an existing order
    mock_storage.get_order.return_value = {"order_id": "ORD_EXISTING"}

    with pytest.raises(ValueError, match="Order with ID 'ORD_EXISTING' already exists."):
        order_tracker.add_order("ORD_EXISTING", "New Item", 1, "CUST001")

def test_add_order_with_zero_quantity_raises_error(order_tracker, mock_storage):
    """Tests that addding an order of zero quantity raises a ValueError. """
    with pytest.raises(ValueError, match="Quantity must be a positive integer."):
        order_tracker.add_order("ORD003", "Keyboard", 0, "CUST001")

def test_add_order_with_negative_quantity_raises_error(order_tracker, mock_storage):
    """Tests that addding an order of negative quantity raises a ValueError."""
    with pytest.raises(ValueError, match="Quantity must be a positive integer."):
        order_tracker.add_order("ORD004", "Monitor", -5, "CUST001")

def test_add_order_with_empty_order_id_raises_error(order_tracker, mock_storage):
    """Test adding a new order with empty order_id raises ValueError. """
    with pytest.raises(ValueError, match="Order ID cannot be empty"):
        order_tracker.add_order("", "Laptop", 1, "CUST001")

def test_add_order_with_empty_item_name_raises_error(order_tracker, mock_storage):
    """Test adding a new order with empty item_name raises ValueError. """
    with pytest.raises(ValueError, match="Item name cannot be empty."):
        order_tracker.add_order("ORD005", "", 1, "CUST001")

def test_add_order_with_empty_customer_id_raises_error(order_tracker, mock_storage):
    """Test adding a new order with empty customer Id raises error. """
    with pytest.raises(ValueError, match="Customer ID cannot be empty"):
        order_tracker.add_order("ORD006", "Laptop", 1, "")

@pytest.mark.parametrize("status", [
    "invalid",
    "unknown",
])
def test_add_order_with_invalid_status_raises_error(order_tracker, mock_storage, status):
    """Test adding order with invalid status raises error. """
    with pytest.raises(ValueError, match="Status must be one of"):
        order_tracker.add_order("ORD007", "Laptop", 1, "CUST001", status=status)

# === get_oder_by_id tests ===
def test_get_order_by_id_existing_order(order_tracker, mock_storage):
    """Tests retrieving an existing order by ID."""
    expected_order = {"order_id": "ORD100", "item_name": "iPad", "quantity": 3}
    mock_storage.get_order.return_value = expected_order

    result = order_tracker.get_order_by_id("ORD100")

    assert result == expected_order
    mock_storage.get_order.assert_called_once_with("ORD100")

def test_get_order_by_id_non_existent_order(order_tracker, mock_storage):
    """Tests retrieving a non existing order returns None. """
    mock_storage.get_order.return_value = None

    result = order_tracker.get_order_by_id("ORD_NONE")

    assert result is None
    mock_storage.get_order.assert_called_once_with("ORD_NONE")

def test_get_order_by_id_empty_id_raises_error(order_tracker, mock_storage):
    """Tests retrieveing an order eith emtpy ID raises ValueError."""
    with pytest.raises(ValueError, match="Order ID cannot be empty"):
        order_tracker.get_order_by_id("")

# === update_order_status Tests ===
def test_update_order_status_successfully(order_tracker, mock_storage):
    """Test successfully updating an order's status. """
    existing_order = {
        "order_id": "ORD200",
        "item_name": "Phone",
        "quantity": 1,
        "customer_id": "CUST200",
        "status": "pending"
    }
    mock_storage.get_order.return_value = existing_order

    order_tracker.update_order_status("ORD200", "shipped")

    mock_storage.save_order.assert_called_once()
    call_args = mock_storage.save_order.call_args[0]

    assert call_args[1]["status"] == "shipped"

def test_update_order_status_with_invalid_status_raises_error(order_tracker, mock_storage):
    """Tests updating to an invalid status raises a valueError. """
    with pytest.raises(ValueError, match="Status must be one of"):
        order_tracker.update_order_status("ORD201", "bad_status")

    mock_storage.get_order.assert_not_called()

def test_update_order_status_non_existing_order_raises_error(order_tracker, mock_storage):
    """Test updating a non-existing order raises a ValueError."""
    mock_storage.get_order.return_value = None

    with pytest.raises(ValueError, match="Order with ID 'ORD_NONE' not found"):
        order_tracker.update_order_status("ORD_NONE", "shipped")

def test_update_order_status_empty_id_raises_error(order_tracker, mock_storage):
    """Test updating an order with empty ID raises a ValueError."""
    with pytest.raises(ValueError, match="Order ID cannot be empty."):
        order_tracker.update_order_status("", "shipped")

# === list_all_order Tests ===
def test_list_all_orders_empty_storage(order_tracker, mock_storage):
    """Tests listing orders when storage is empty"""
    mock_storage.get_all_orders.return_value = {}

    result = order_tracker.list_all_orders()

    assert result == []
    mock_storage.get_all_orders.assert_called_once()

def test_list_all_orders_with_multiple_orders(order_tracker, mock_storage):
    """Tests listing all orders when multiple orders exist."""
    orders_dict = {
        "ORD300": {"order_id": "ORD300", "item_name": "Laptop", "status": "pending"},
        "ORD301": {"order_id": "ORD301", "item_name": "Mouse", "status": "shipped"},
        "ORD302": {"order_id": "ORD302", "item_name": "Keyboard", "status": "delivered"}
    }

    mock_storage.get_all_orders.return_value = orders_dict

    result = order_tracker.list_all_orders()

    assert len(result) == 3
    result_ids = {order["order_id"] for order in result}
    assert result_ids == {"ORD300", "ORD301", "ORD302"}

# === list_orders_by_status Tests ===
def test_list_orders_by_status_with_mateches(order_tracker, mock_storage):
    """Test listing orders by status when orders match."""
    order_dict = {
        "ORD400": {"order_id": "ORD400", "item_name": "Laptop", "status": "pending"},
        "ORD401": {"order_id": "ORD401", "item_name": "Mouse", "status": "shipped"},
        "ORD402": {"order_id": "ORD402", "item_name": "Keyboard", "status": "pending"}
    }

    mock_storage.get_all_orders.return_value = order_dict

    result = order_tracker.list_orders_by_status("pending")

    assert len(result) == 2
    result_ids = {order["order_id"] for order in result}
    assert result_ids == {"ORD400", "ORD402"}

def test_list_orders_by_status_with_no_matches(order_tracker, mock_storage):
    """Test listing orders by status when no match."""
    orders_dict = {
        "ORD500": {"order_id": "ORD500", "item_name": "Laptop", "status": "shipped"},
        "ORD501": {"order_id": "ORD501", "item_name": "Mouse", "status": "delivered"}
    }
    mock_storage.get_all_orders.return_value = orders_dict

    result = order_tracker.list_orders_by_status("pending")

    assert result == []

def test_list_orders_by_status_empty_storage(order_tracker, mock_storage):
    """Tests listing orders by status when storage is empty."""
    mock_storage.get_all_orders.return_value = {}

    result = order_tracker.list_orders_by_status("pending")
    assert result == []

def test_list_orders_by_status_raises_error(order_tracker, mock_storage):
    """Test the listing orders with empty status raises a ValueError."""
    with pytest.raises(ValueError, match="Status cannot be empty."):
        order_tracker.list_orders_by_status("")

def test_list_orders_by_status_invalid_status_raises_error(order_tracker, mock_storage):
    """Tests listing orders with invalid status raises a ValueError."""
    with pytest.raises(ValueError, match="Status must be one of"):
        order_tracker.list_orders_by_status("invalid_status")

# === delete_order Tests ===
def test_delete_order_successfully(order_tracker, mock_storage):
    """Tests successfully deleting an order."""
    existing_order = {
        "order_id": "ORD600",
        "item_name": "Laptop",
        "quantity": 1,
        "customer_id": "CUST600",
        "status": "pending"
    }

    mock_storage.get_order.return_value = existing_order

    result = order_tracker.delete_order("ORD600")

    assert result == existing_order
    mock_storage.delete_order.assert_called_once_with("ORD600")

def test_delete_order_non_existing_raises_error(order_tracker, mock_storage):
    """Tests deleting a non-existent order raises a ValueError."""
    mock_storage.get_order.return_value = None

    with pytest.raises(ValueError, match="Order with ID 'ORD_NONE' not found."):
        order_tracker.delete_order("ORD_NONE")

def test_delete_order_empty_id_raises_error(order_tracker, mock_storage):
    """Tests deleting an order with empty ID raises a ValueError"""
    with pytest.raises(ValueError, match="Order ID cannot be empty."):
        order_tracker.delete_order("")