from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from app.main import register_routes


# Fake components
class FakeModel:
    def __init__(self):
        self.classes_ = ["No", "Yes"]

    def predict(self, df):
        # return probabilities to simulate a real model's output
        return np.array([[0.2, 0.8]])  # high probability for "Yes"


class FakeRequest(BaseModel):
    gender: str
    tenure: int
    MonthlyCharges: float


# create test app with fake components
def create_test_app():
    app = FastAPI()

    # mock state
    app.state.model = FakeModel()
    app.state.RequestModel = FakeRequest
    app.state.threshold = 0.5

    # real route registration
    register_routes(app)

    return app


# tests
def test_health_check():
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    app = create_test_app()
    client = TestClient(app)

    sample_input = {
        "gender": "Male",
        "tenure": 10,
        "MonthlyCharges": 50.0
    }

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Yes", "No"]