# Customer Churn Prediction System (MLOps)

End-to-end machine learning system for customer churn prediction, 
featuring model versioning with MLflow and real-time inference via FastAPI.

## Architecture

FastAPI → MLflow Model Registry → Stored Models (mlruns)

- FastAPI serves prediction API
- MLflow manages model versioning (Champion / Latest)
- Docker ensures reproducible deployment

### Features

- Dynamic model loading from MLflow (Champion alias fallback)
- Schema-driven request validation (Pydantic dynamic model)
- Probability-based prediction with configurable threshold
- Dockerized deployment (API + MLflow)
- Retry mechanism for service readiness (MLflow dependency)

### Tech Stack

- Python
- FastAPI
- MLflow
- scikit-learn
- Docker / Docker Compose
- AWS EC2 (for deployment demo)

### Example Request

POST /predict

```json
{
  "gender": "Male",
  "tenure": 12,
  "monthly_charges": 70.5
}
```

### Example Response

``` json
{
  "prediction": "Yes",
  "probability": 0.82,
  "threshold": 0.5
}
```

## Project Structure

``` bash
src/
app/
main.py
mlruns/
docker-compose.yml
Dockerfile
```

## Docker Deployment

```bash
git clone https://github.com/yeeislazy/churn_detection
cd churn_detection

docker compose up --build
```

Services:

- API → <http://localhost:8000>
- MLflow → <http://localhost:5000>

## Other features

- download dataset from Kaggle

```bash
# requires Kaggle API credentials
uv run download-data
```

- preprocess data

```bash
uv run preprocess-data
```

- train model and log to MLflow

```bash
uv run train-model
```
