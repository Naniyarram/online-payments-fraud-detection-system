from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

payload = {
    "transaction": {
        "trans_date_trans_time": "2019-01-01 12:00:00",
        "cc_num": 123456789,
        "merchant": "fraud_shoes_inc",
        "category": "shopping_net",
        "amt": 950.00,
        "first": "Alice",
        "last": "Smith",
        "gender": "F",
        "street": "123 Broadway",
        "city": "New York",
        "state": "NY",
        "zip": 10002,
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8500000,
        "job": "Designer",
        "dob": "1990-05-15",
        "trans_num": "tx_abc123",
        "unix_time": 1546344000,
        "merch_lat": 40.75,
        "merch_long": -73.99
    }
}

# The client ContextManager startup executes on_event('startup') triggering artifact loads
with client as cl:
    print("Sending post request to /explain...")
    response = cl.post("/explain", json=payload)
    print(f"Response status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("\n=== GENERATED FORENSIC ANALYST REPORT ===")
        print(data["analyst_report"])
        print("=========================================")
    else:
        print(response.text)
