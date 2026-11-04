from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_prediction_endpoint():
    payload = {
        "customer_data": {
            "age": 35,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.1,
            "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4,
            "euribor3m": 4.857,
            "nr.employed": 5191.0
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] in ["yes", "no"]
    assert 0 <= response.json()["subscription_probability"] <= 1


def test_prediction_with_missing_features():
    response = client.post(
        "/predict",
        json={"customer_data": {"age": 35}}
    )

    assert response.status_code == 400