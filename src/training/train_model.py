import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import mlflow
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from pathlib import Path
import re
import json
import argparse

def find_best_threshold(y_true, y_proba):
    best_t = 0
    best_f1 = 0

    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

def build_fit_pipeline(schema, model,Y_train, X_train):
    cat_columns = [col["name"] for col in schema["features"]["categorical"]]
    num_columns = [col["name"] for col in schema["features"]["numerical"]]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_columns),
        ]
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, Y_train)
        
    return pipeline

def evaluate_model(model, Y_test, X_test):
    classes = model.classes_
    pos_index = list(classes).index("Yes")
    
    y_proba = model.predict_proba(X_test)[:, pos_index]
    
    Y_test_binary = (Y_test == "Yes").astype(int)
    best_t, _ = find_best_threshold(Y_test_binary, y_proba)
    ypred = (y_proba >= best_t).astype(int)
    
    print(classification_report(Y_test_binary, ypred))
    class_report_dict = classification_report(Y_test_binary, ypred, output_dict=True)
    return class_report_dict , best_t 

def mlflow_log(model,Y_train, X_train, parameters, mlflow_experiment, class_report_dict, version, dataset_name, schema):
    
    sign = infer_signature(X_train,model.predict(X_train))
    
    if sign.outputs.inputs[0].type == 'object':
        new_output = Schema([ColSpec("string")])

        signature = ModelSignature(
            inputs=sign.inputs,
            outputs=new_output
        )
    else:
        signature = sign
    train_ds = mlflow.data.from_pandas(
        X_train.assign(Churn=Y_train),
        name=dataset_name + ':' + version
    )
    
    num_columns = X_train.select_dtypes(include='number').columns
    num_columns = num_columns.drop("SeniorCitizen")
    cat_columns = X_train.columns.drop(num_columns)
    schema["threshold"]= parameters.get("best_threshold", 0.5)


    
    with mlflow.start_run(experiment_id=mlflow_experiment.experiment_id):
        mlflow.log_params(parameters)
        mlflow.log_param("data_version", version)
        
        mlflow.log_metric("accuracy", class_report_dict["accuracy"])
        mlflow.log_metric("precision", class_report_dict["1"]["precision"])
        mlflow.log_metric("recall", class_report_dict["1"]["recall"])
        mlflow.log_metric("f1-score", class_report_dict["1"]["f1-score"])
        
        mlflow.log_input(train_ds, context="training")
        
        mlflow.sklearn.log_model(model, parameters['model'], signature=signature, pyfunc_predict_fn="predict_proba")
        
        mlflow.log_dict(schema, "schema.json")
    
def ml_pipeline(schema, model, Y_train, X_train, X_test, Y_test, parameters, mlflow_experiment, version, dataset_name):
    parameters['model'] = model.__class__.__name__
    model = build_fit_pipeline(schema, model, Y_train, X_train)
    class_report_dict , parameters['best_threshold'] = evaluate_model(model, Y_test, X_test)
    mlflow_log(model, Y_train, X_train, parameters, mlflow_experiment, class_report_dict, version, dataset_name,schema)


def register_best_model(mlflow_experiment, metric):
    all_runs_df = mlflow.search_runs(experiment_ids=[mlflow_experiment.experiment_id], order_by=[f"metrics.{metric} DESC"])
    best_model = all_runs_df.iloc[0]
    best_run_id = best_model['run_id']
    model_name = best_model['params.model']
    mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/{model_name}", 
        name="customer-churn-model"
        )
    print(f"Best model registered: {model_name} from run {best_run_id} with {metric} = {best_model[f'metrics.{metric}']}")

def main():
    parser = argparse.ArgumentParser(description="Run model experiments for churn detection")
    parser.add_argument('--dataset-version', type=str, help='Specify the dataset version to use (e.g., v1, v2). If not provided, the latest version will be used.')
    parser.add_argument('--experiment-name', type=str, default="churn_detection", help='Specify the MLflow experiment name. Default is "churn_detection".')
    parser.add_argument('--track-uri', type=str, help='Specify the MLflow tracking URI. Default is "http://localhost:5000".')
    parser.add_argument('--metric', type=str, default="recall", choices=["f1", "precision", "recall", "accuracy"], help='Specify the metric to optimize for model registration. Default is "recall".')
    
    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent.parent
    dataset_dir = root_dir / "data" / "processed"
    versions = set([v.name for v in dataset_dir.iterdir() if re.match(r"v\d+", v.name)])
    
    if args.dataset_version:
        if args.dataset_version in versions:
            version = args.dataset_version
        else:
            print(f"Specified dataset version {args.dataset_version} not found. Available versions: {versions}")
            return            
    else:
        version = sorted(versions)[-1] if versions else None
    
    dataset_dir = dataset_dir / version
    dataset_name = 'telco-customer-churn'

    trainset = pd.read_csv(dataset_dir / "train.csv")
    X_train = trainset.drop("Churn", axis=1)
    Y_train = trainset["Churn"]

    testset = pd.read_csv(dataset_dir / "test.csv")
    X_test = testset.drop("Churn", axis=1)
    Y_test = testset["Churn"]

    schema = json.load(open(dataset_dir / "schema.json", "r"))

    if os.getenv("MLFLOW_TRACKING_URI"):
        track_uri = os.getenv("MLFLOW_TRACKING_URI")
    elif args.track_uri:
        track_uri = args.track_uri  
    else:
        track_uri = "http://localhost:5000"

    mlflow.set_tracking_uri(track_uri)
    exp = mlflow.set_experiment(args.experiment_name)

    try_parameters = {
        "n_estimators": [50, 100, 200,500],
        "class_weight": ["balanced", None],
        "random_state": 42
    }

    for n_est in try_parameters["n_estimators"]:
        for cw in try_parameters["class_weight"]:
            parameters = {
                "n_estimators": n_est,
                "class_weight": cw,
                "random_state": try_parameters["random_state"]
            }
            model = RandomForestClassifier(**parameters)
            ml_pipeline(schema, model, Y_train, X_train, X_test, Y_test, parameters, exp, version, dataset_name)

    try_parameters = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    for C in try_parameters['C']:
        for penalty in try_parameters['penalty']:
            if penalty == 'l1':
                solver = 'liblinear'
            else:
                solver = 'lbfgs'
            parameters = {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'max_iter': 1000
            }
            model = LogisticRegression(**parameters)
            ml_pipeline(schema, model, Y_train, X_train, X_test, Y_test, parameters, exp, version, dataset_name)
            
    register_best_model(exp, args.metric)

if __name__ == "__main__":
    main()