import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import json

def main():
    root_dir = Path(__file__).parent.parent.parent
    raw_dir = root_dir / "data" / "raw"
    df = pd.read_csv(raw_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.head()

    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.isna().sum()

    df = df.dropna()

    num_columns = df.select_dtypes(include='number').columns
    num_columns = num_columns.drop("SeniorCitizen")

    X = df.drop("Churn", axis=1)
    Y = df["Churn"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    trainset = pd.concat([X_train, Y_train], axis=1)
    testset = pd.concat([X_test, Y_test], axis=1)

    versions = sorted([v.name for v in Path(root_dir / "data"/"processed").iterdir() if re.match(r"v\d+", v.name)])
    if versions:
        latest_version = versions[-1]
        new_version = f"v{int(latest_version[1:]) + 1}"
    else:
        new_version = "v1"

    processed_dir = root_dir / "data" / "processed" / new_version
    processed_dir.mkdir(parents=True, exist_ok=True)

    trainset.to_csv(processed_dir / "train.csv", index=False)
    testset.to_csv(processed_dir / "test.csv", index=False)

    cat_columns = X.columns.drop(num_columns).to_list()
    cat_columns.append("SeniorCitizen")
    schema = {
        "features" : {
            "numerical": [],
            "categorical": []
        },
        "target": ["Churn"],
        "feature_engineering": {
            "numerical": {
            "scaler": "StandardScaler"
            },
            "categorical": {
            "encoding": "OneHotEncoder",
            "handle_unknown": "ignore"
            }
        }
    }
    for col in num_columns:
        name = col
        type = df[col].dtype
        schema["features"]["numerical"].append({"name": name, "type": str(type)})
        
    for col in cat_columns:
        name = col
        values = df[col].unique().tolist()
        schema["features"]["categorical"].append({"name": name, "values": values})

    with open(processed_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=4)

if __name__ == "__main__":
    main()
