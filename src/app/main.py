from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import mlflow
from mlflow.tracking import MlflowClient
import json
import tempfile
import pandas as pd
import uvicorn
from pydantic import create_model
from enum import Enum
import os
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# file handler
log_file = Path(__file__).parent.parent.parent / "logs" / "app.log"
log_file.parent.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

MODEL_NAME = "customer-churn-model"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =========================
# dynamic schema → pydantic
# =========================
def create_request_model(schema):
    logger.info("Creating request model from schema...")
    fields = {}

    for col in schema["features"]["numerical"]:
        if 'int' in col["type"]:
            fields[col["name"]] = (int, ...)
        elif 'float' in col["type"]:
            fields[col["name"]] = (float, ...)

    for col in schema["features"]["categorical"]:
        name = col["name"]
        values = col.get("values")
        if values:
            enum_cls = Enum(
                name,
                {f"{name}_{v}": v for v in values}
            )
            fields[name] = (enum_cls, ...)
        else:
            fields[name] = (str, ...)
    try:
        model = create_model("ChurnRequest", **fields)
        logger.info("Request model created successfully")
        return model
    except Exception as e:
        logger.error("Error creating request model", exc_info=True)
        raise e


# dynamic routes
def register_routes(app: FastAPI):

    Model = app.state.RequestModel

    @app.post("/predict")
    async def predict(data: Model):

        data_dict = {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in data.model_dump().items()
        }
        df = pd.DataFrame([data_dict])

        model = app.state.model
        threshold = app.state.threshold

        # determine positive class index
        try:
            classes = model.classes_
            pos_index = list(classes).index("Yes")
            logger.info('Positive class index found at: %d', pos_index)
        except Exception:
            pos_index = 1
            logger.warning('Positive class index not found, defaulting to index 1')
            
        # predict
        try:
            prob = float(model.predict(df)[:, pos_index][0])
            pred = int(prob >= threshold)
            logger.info(f"Predicted probability: {prob}, Threshold: {threshold}, Prediction: {pred}")
        except Exception as e:
            logger.error("Error during prediction", exc_info=True)
            raise HTTPException(status_code=500, detail="Prediction error")

        return {
            "prediction": "Yes" if pred else "No",
            "probability": prob,
            "threshold": threshold
        }

    @app.get("/")
    async def health_check():
        return {"status": "ok"}

# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")

    client = MlflowClient()

    model = None
    version = None
    
    for i in range(10):
        try:
            logger.info(f"Trying to connect MLflow... attempt {i+1}")
            model_uri = f"models:/{MODEL_NAME}@champion"
            model = mlflow.pyfunc.load_model(model_uri)
            version = client.get_model_version_by_alias(MODEL_NAME, "champion")
            logger.info("Loaded CHAMPION model")
            break

        except Exception as e:
            logger.warning("Retrying MLflow...", e)
            time.sleep(2)
        
    if model is None:
        for i in range(5):
            try:
                logger.info("No champion found, fallback to latest")
                versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                latest = sorted(versions, key=lambda x: int(x.version))[-1]

                model_uri = f"models:/{MODEL_NAME}/{latest.version}"
                model = mlflow.pyfunc.load_model(model_uri)
                version = latest
                logger.info(f"Loaded latest model version {latest.version}")
                break
            except Exception as e:
                logger.warning("Retrying MLflow...", e)
                time.sleep(2)

    if model is None:
        logger.error("Failed to load model from MLflow after multiple attempts")
        raise RuntimeError("Failed to connect to MLflow")
    
    # load schema
    run_id = version.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = client.download_artifacts(run_id, "schema.json", dst_path=tmpdir)
        with open(schema_path) as f:
            schema = json.load(f)

    # store in app state for global access
    app.state.model = model
    app.state.schema = schema
    app.state.threshold = schema.get("threshold", 0.5)
    app.state.RequestModel = create_request_model(schema)

    # register routes after model is loaded
    register_routes(app)

    logger.info("Model loaded")

    yield

    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()