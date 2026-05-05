from fastapi.testclient import TestClient
from app.main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_prediction():
    sample_input = {
        'gender': 'Male',
        'SeniorCitizen': 1,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 61,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 25.0,
        'TotalCharges': 1501.75
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=sample_input)

        assert response.status_code == 200
        assert "prediction" in response.json()