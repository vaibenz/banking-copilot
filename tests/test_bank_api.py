from fastapi.testclient import TestClient

from copilot.bank_api import STORE, app

client = TestClient(app)


def test_accounts_scoped_to_customer():
    ids = {a["account_id"] for a in client.get("/customers/C001/accounts").json()["accounts"]}
    assert ids == {"SAV-001", "CUR-002"}


def test_cannot_freeze_another_customers_card():
    assert client.post("/customers/C001/cards/5530/freeze").status_code == 404
    assert STORE["cards"]["5530"]["status"] == "active"


def test_dispute_rules():
    assert client.post("/customers/C001/disputes", json={"transaction_id": "TXN-2001", "reason": "not mine"}).status_code == 404
    assert client.post("/customers/C001/disputes", json={"transaction_id": "TXN-1006", "reason": "old"}).status_code == 422
    assert client.post("/customers/C001/disputes", json={"transaction_id": "TXN-1004", "reason": "credit"}).status_code == 422
    assert client.post("/customers/C001/disputes", json={"transaction_id": "TXN-1003", "reason": "fraud"}).status_code == 201
    assert client.post("/customers/C001/disputes", json={"transaction_id": "TXN-1003", "reason": "again"}).status_code == 409
