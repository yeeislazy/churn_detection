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

MODEL_NAME = "customer-churn-model"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    print("MLFLOW_TRACKING_URI not set, using local as default")
    MLFLOW_TRACKING_URI = (Path(__file__).parent.parent.parent / "mlruns").as_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =========================
# dynamic schema → pydantic
# =========================
def create_request_model(schema):
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

    return create_model("ChurnRequest", **fields)


# =========================
# 动态注册 endpoint（关键）
# =========================
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

        # predict
        try:
            classes = model.classes_
            pos_index = list(classes).index("Yes")
        except Exception:
            pos_index = 1

        prob = float(model.predict(df)[:, pos_index][0])
        pred = int(prob >= threshold)

        return {
            "prediction": "Yes" if pred else "No",
            "probability": prob,
            "threshold": threshold
        }


# =========================
# lifespan
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")

    client = MlflowClient()

    model = None
    version = None
    
    for i in range(10):
        try:
            print(f"Trying to connect MLflow... attempt {i+1}")
            model_uri = f"models:/{MODEL_NAME}@champion"
            model = mlflow.pyfunc.load_model(model_uri)
            version = client.get_model_version_by_alias(MODEL_NAME, "champion")
            print("Loaded CHAMPION model")
            break

        except Exception as e:
            print("Retrying MLflow...", e)
            time.sleep(2)
        
    if model is None:
        for i in range(5):
            try:
                print("No champion found, fallback to latest")
                versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                latest = sorted(versions, key=lambda x: int(x.version))[-1]

                model_uri = f"models:/{MODEL_NAME}/{latest.version}"
                model = mlflow.pyfunc.load_model(model_uri)
                version = latest
                print(f"Loaded latest model version {latest.version}")
                break
            except Exception as e:
                print("Retrying MLflow...", e)
                time.sleep(2)

    if model is None:
        raise RuntimeError("Failed to connect to MLflow")
    
    # load schema
    run_id = version.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_path = client.download_artifacts(run_id, "schema.json", dst_path=tmpdir)
        with open(schema_path) as f:
            schema = json.load(f)

    # 存 state
    app.state.model = model
    app.state.schema = schema
    app.state.threshold = schema.get("threshold", 0.5)
    app.state.RequestModel = create_request_model(schema)

    # 🔥 关键：这里注册 routes
    register_routes(app)

    print("Model loaded")

    yield

    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()