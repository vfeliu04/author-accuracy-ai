import pytest

from app import create_app


@pytest.fixture()
def client(settings):
    app = create_app()
    app.config.update({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_protected_endpoint_requires_key(client):
    response = client.get("/api/claims")
    assert response.status_code == 401
    response = client.get("/api/claims", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert "claims" in response.json
