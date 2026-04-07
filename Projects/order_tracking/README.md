# Udatracker Starter Code

This directory contains the starter code for the Udatracker project. The initial structure of directories and files is described below.

### Project Structure

```
.
├── backend
│   ├── __init__.py
│   ├── app.py                      # Flask application
│   ├── in_memory_storage.py        # In-memory storage implementation
│   ├── order_tracker.py            # Business logic for order tracking
│   ├── requirements.txt
│   └── tests
│       ├── __init__.py
│       ├── test_api.py             # API integration tests
│       └── test_order_tracker.py   # Unit tests for order_tracker.py
├── frontend
│   ├── css
│   │   └── style.css
│   ├── index.html
│   └── js
│       └── script.js
├── utils
│   ├── logs.py                   # Logging utilities 
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── pytest.ini
└── README.md
```
# Order Tracker REST API

A production-ready order management system built with Flask, following Test-Driven Development (TDD) principles.

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Running with Docker](#running-with-docker)
- [Development](#development)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Features

✅ **Complete CRUD Operations** - Create, Read, Update, Delete orders  
✅ **Status Management** - Track order lifecycle (pending → processing → shipped → delivered)  
✅ **Filtering & Search** - Filter orders by status  
✅ **Input Validation** - Comprehensive validation at multiple layers  
✅ **Error Handling** - Proper HTTP status codes and error messages  
✅ **Test Coverage** - 35+ unit tests with TDD approach  
✅ **Docker Support** - Containerized for easy deployment  
✅ **RESTful Design** - Follows REST API best practices  

---

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd order-tracker

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

Server will start on `http://localhost:5000`

### Using Docker

```bash
# Build and run
docker-compose up --build

# Or build manually
docker build -t order-tracker .
docker run -p 5000:5000 order-tracker
```

---

## API Reference

Base URL: `http://localhost:5000`

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/orders` | Create a new order |
| GET | `/api/orders/<order_id>` | Get order by ID |
| PUT | `/api/orders/<order_id>/status` | Update order status |
| GET | `/api/orders` | List all orders |
| GET | `/api/orders?status=<status>` | Filter orders by status |
| DELETE | `/api/orders/<order_id>` | Delete an order |

---

### 1. Create Order

**Endpoint:** `POST /api/orders`

**Request Body:**
```json
{
  "order_id": "ORD001",
  "item_name": "Laptop",
  "quantity": 2,
  "customer_id": "CUST001",
  "status": "pending"  // optional, defaults to "pending"
}
```

**Response:** `201 Created`
```json
{
  "message": "Order added successfully",
  "order_id": "ORD001"
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/api/orders \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "ORD001",
    "item_name": "Laptop",
    "quantity": 2,
    "customer_id": "CUST001"
  }'
```

**Validation Rules:**
- `order_id`: Required, non-empty, must be unique
- `item_name`: Required, non-empty
- `quantity`: Required, positive integer
- `customer_id`: Required, non-empty
- `status`: Optional, must be one of: `pending`, `processing`, `shipped`, `delivered`, `cancelled`

---

### 2. Get Order

**Endpoint:** `GET /api/orders/<order_id>`

**Response:** `200 OK`
```json
{
  "order_id": "ORD001",
  "item_name": "Laptop",
  "quantity": 2,
  "customer_id": "CUST001",
  "status": "pending"
}
```

**cURL Example:**
```bash
curl http://localhost:5000/api/orders/ORD001
```

**Error Response:** `404 Not Found`
```json
{
  "error": "Order with ID 'ORD999' not found"
}
```

---

### 3. Update Order Status

**Endpoint:** `PUT /api/orders/<order_id>/status`

**Request Body:**
```json
{
  "status": "shipped"
}
```

**Response:** `200 OK`
```json
{
  "message": "Order status updated successfully",
  "order_id": "ORD001",
  "new_status": "shipped"
}
```

**cURL Example:**
```bash
curl -X PUT http://localhost:5000/api/orders/ORD001/status \
  -H "Content-Type: application/json" \
  -d '{"status": "shipped"}'
```

**Valid Statuses:**
- `pending` - Initial state
- `processing` - Order is being prepared
- `shipped` - Order has been shipped
- `delivered` - Order delivered to customer
- `cancelled` - Order cancelled

---

### 4. List All Orders

**Endpoint:** `GET /api/orders`

**Response:** `200 OK`
```json
[
  {
    "order_id": "ORD001",
    "item_name": "Laptop",
    "quantity": 2,
    "customer_id": "CUST001",
    "status": "shipped"
  },
  {
    "order_id": "ORD002",
    "item_name": "Mouse",
    "quantity": 5,
    "customer_id": "CUST002",
    "status": "pending"
  }
]
```

**cURL Example:**
```bash
curl http://localhost:5000/api/orders
```

---

### 5. Filter Orders by Status

**Endpoint:** `GET /api/orders?status=<status>`

**Response:** `200 OK`
```json
[
  {
    "order_id": "ORD001",
    "item_name": "Laptop",
    "quantity": 2,
    "customer_id": "CUST001",
    "status": "shipped"
  }
]
```

**cURL Examples:**
```bash
# Get all pending orders
curl "http://localhost:5000/api/orders?status=pending"

# Get all shipped orders
curl "http://localhost:5000/api/orders?status=shipped"

# Get all delivered orders
curl "http://localhost:5000/api/orders?status=delivered"
```

---

### 6. Delete Order

**Endpoint:** `DELETE /api/orders/<order_id>`

**Response:** `200 OK`
```json
{
  "message": "Order deleted successfully",
  "deleted_order": {
    "order_id": "ORD001",
    "item_name": "Laptop",
    "quantity": 2,
    "customer_id": "CUST001",
    "status": "pending"
  }
}
```

**cURL Example:**
```bash
curl -X DELETE http://localhost:5000/api/orders/ORD001
```

**Error Response:** `404 Not Found`
```json
{
  "error": "Order with ID 'ORD999' not found."
}
```

---

## HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful GET/PUT/DELETE |
| 201 | Created | Successfully created new order |
| 400 | Bad Request | Validation error, invalid input |
| 404 | Not Found | Order doesn't exist |
| 500 | Internal Server Error | Unexpected server error |

---

## Running with Docker

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop the service
docker-compose down

# View logs
docker-compose logs -f
```

### Using Docker Directly

```bash
# Build the image
docker build -t order-tracker:latest .

# Run the container
docker run -p 5000:5000 order-tracker:latest

# Run with environment variables
docker run -p 5000:5000 \
  -e FLASK_ENV=development \
  order-tracker:latest
```

### Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Flask environment (development/production) |
| `PORT` | `5000` | Port to bind the application |

---

### Running Tests

```bash
# Run unit tests
pytest backend/tests/test_order_tracker.py -v

# Run API integration tests (requires server running)
python test_api.py

# Run with coverage
pytest --cov=backend backend/tests/
```

### Code Quality

```bash
# Format code
black backend/

# Lint code
pylint backend/

# Type checking
mypy backend/
```

---

## Testing

### Automated Test Suite

The project includes comprehensive test coverage:

**Unit Tests (35+ tests):**
- `add_order`: 9 tests (validation, duplicates, edge cases)
- `get_order_by_id`: 3 tests (existing, non-existent, validation)
- `update_order_status`: 4 tests (success, validation, errors)
- `list_all_orders`: 2 tests (empty, multiple)
- `list_orders_by_status`: 5 tests (filtering, validation)
- `delete_order`: 3 tests (success, not found, validation)

**API Integration Tests:**
- Complete endpoint testing
- Error scenario validation
- Status code verification

### Manual Testing Examples

**Complete Order Lifecycle:**

```bash
# 1. Create order
curl -X POST http://localhost:5000/api/orders \
  -H "Content-Type: application/json" \
  -d '{"order_id":"ORD001","item_name":"Laptop","quantity":2,"customer_id":"CUST001"}'

# 2. Get order details
curl http://localhost:5000/api/orders/ORD001

# 3. Update to processing
curl -X PUT http://localhost:5000/api/orders/ORD001/status \
  -H "Content-Type: application/json" \
  -d '{"status":"processing"}'

# 4. Update to shipped
curl -X PUT http://localhost:5000/api/orders/ORD001/status \
  -H "Content-Type: application/json" \
  -d '{"status":"shipped"}'

# 5. List all orders
curl http://localhost:5000/api/orders

# 6. Filter shipped orders
curl "http://localhost:5000/api/orders?status=shipped"

# 7. Delete order
curl -X DELETE http://localhost:5000/api/orders/ORD001
```

---

## Common Error Scenarios

### Duplicate Order ID
```bash
curl -X POST http://localhost:5000/api/orders \
  -H "Content-Type: application/json" \
  -d '{"order_id":"ORD001","item_name":"Laptop","quantity":2,"customer_id":"CUST001"}'

# Response: {"error": "Order with ID 'ORD001' already exists."}
```

### Invalid Quantity
```bash
curl -X POST http://localhost:5000/api/orders \
  -H "Content-Type: application/json" \
  -d '{"order_id":"ORD002","item_name":"Mouse","quantity":0,"customer_id":"CUST002"}'

# Response: {"error": "Quantity must be a positive integer."}
```

### Order Not Found
```bash
curl http://localhost:5000/api/orders/NONEXISTENT

# Response: {"error": "Order with ID 'NONEXISTENT' not found"}
```

### Invalid Status
```bash
curl -X PUT http://localhost:5000/api/orders/ORD001/status \
  -H "Content-Type: application/json" \
  -d '{"status":"invalid_status"}'

# Response: {"error": "Status must be one of {...}"}
```

---

## Production Considerations

### Security
- Add authentication (JWT/OAuth)
- Implement rate limiting
- Enable CORS with proper origins
- Use HTTPS in production
- Sanitize all inputs
- Implement API versioning

### Performance
- Add caching (Redis)
- Implement pagination for list endpoints
- Add database indexing
- Use connection pooling
- Enable compression

### Monitoring
- Add health check endpoint
- Implement logging aggregation
- Set up error tracking (Sentry)
- Add metrics collection (Prometheus)
- Monitor API performance

### Database
- Replace InMemoryStorage with PostgreSQL/MySQL
- Implement database migrations
- Add transaction support
- Implement soft deletes

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Implement your changes
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

## Acknowledgments

Built with:
- Flask - Web framework
- pytest - Testing framework
- Docker - Containerization

Developed following TDD principles and REST API best practices.