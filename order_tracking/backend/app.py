from flask import Flask, request, jsonify, send_from_directory
from backend.order_tracker import OrderTracker
from backend.in_memory_storage import InMemoryStorage
from utils import logger

app = Flask(__name__, static_folder='../frontend')
in_memory_storage = InMemoryStorage()
order_tracker = OrderTracker(in_memory_storage)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "order_tracker",
        "version": "1.0.0"
    }), 200
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/api/orders', methods=['POST'])
def add_order_api():
    """
    Add a new order.
    Expected JSON body:
    {
        "order_id": "string",
        "item_name": "string",
        "quantity": integer,
        "customer_id": "string",
        "status": "string" (optional, defaults to "pending")
    }
    """
    try:
        data = request.get_json()
        
        # validation
        required_fields = ['order_id', 'item_name', 'quantity', 'customer_id']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            logger.error(f"[add_order] Missing required fields: {missing_fields}")
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 404
        
        # extracts fields
        order_id = data['order_id']
        item_name = data['item_name']
        quantity = data['quantity']
        customer_id = data['customer_id']
        status = data.get('status', 'pending')  # optional, defaults to "pending"

        if not isinstance(quantity, int):
            logger.error(f"[add_order] Invalid quantity type: {type(quantity)}")
            return jsonify({
                "error": "Quantity must be an integer"
            }), 400
        
        order_tracker.add_order(order_id, item_name, quantity, customer_id, status)

        logger.info(f"[add_order] Order {order_id} added successfully.")
        return jsonify({
            "message": "order added successfully",
            "order_id": order_id
        }), 201
        
    except ValueError as e:
        logger.error(f"[add_order] Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"[add_order] Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/orders/<string:order_id>', methods=['GET'])
def get_order_api(order_id):
    """
    Get order details by ID.
    Returns 404 if order not found.
    """
    try:
        order = order_tracker.get_order_by_id(order_id)

        if order is None:
            logger.warning(f"[get_order] Order {order_id} not found")
            return jsonify({"error": f"Order with ID '{order_id} not found"}), 404
        
        logger.info(f"[get_order] Order {order_id} retrieved successfully.")
        return jsonify(order), 200
    
    except ValueError as e:
        logger.error(f"[get_order] Unexpected error: {e}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"[get_order] Unexpected error: {e}")
        return jsonify({"error": "internal server error"}), 500

@app.route('/api/orders/<string:order_id>/status', methods=['PUT'])
def update_order_status_api(order_id):
    """
    Update order status.
    Expected JSON body:
    {
        "status": "string"
    }
    """
    try:
        data = request.get_json()

        if 'status' not in data:
            logger.warning(f"[update_order] Missing 'status' field")
            return jsonify({"error": "Missing required field: status"}), 404
        
        new_status = data['status']

        order_tracker.update_order_status(order_id, new_status)

        logger.info(f"[update_order] Order {order_id} status updated to {new_status}")
        return jsonify({
            "message": "Order status update successfully",
            "order_id": order_id,
            "new_status": new_status
        }), 200

    except ValueError as e:
        logger.error(f"[update_order] Validation error: {e}")
        if "not found" in str(e).lower():
            return jsonify({"error": str(e)}), 400
        else:
            return jsonify({"error": str(e)}), 400
        
    except Exception as e:
        logger.error(f"[update_order] Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/orders', methods=['GET'])
def list_orders_api():
    """
    List all orders or filter by status.
    Query parameter:
    - status: (optional) filter orders by status
    """
    try:
        status_filter = request.args.get('status')

        if status_filter:
            orders = order_tracker.list_orders_by_status(status_filter)
            logger.info(f"[list_orders] Retrieved all {len(orders)} orders with status '{status_filter}'")
        else:
            orders = order_tracker.list_all_orders()
            logger.info(f"Retrieved all {len(orders)} orders")
        
        return jsonify(orders), 200
    
    except ValueError as e:
        logger.error(f"[list_orders] Validation error: {e}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"[list_orders] Unexpected error: {e}")
        return jsonify({"error": "Internal server erro"}), 500
    
@app.route('/api/orders/<string:order_id>', methods=['DELETE'])
def delete_order_api(order_id):
    """
    Delete an order by ID.
    Returns 200 with the deleted order details.
    """
    try:
        deleted_order = order_tracker.delete_order(order_id)

        logger.info(f"Order {order_id} deleted successuflly")
        return jsonify({
            "message": "Order deleted successfully",
            "deleted_order": deleted_order
        }), 200
    except ValueError as e:
        logger.error(f"[delete_order] Validation error: {e}")

        if "not found" in str(e).lower():
            return jsonify({"error": str(e)}), 400
        else:
            return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[delete_order] Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
