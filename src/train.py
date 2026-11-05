import os
import yaml
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

df = pd.read_csv("data/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("TP_Iris_Classification")

# Dataset avec le chemin source pour lier le fichier
dataset = mlflow.data.from_pandas(df, source="data/raw/iris.csv", targets="target", name="iris_dataset")

models = {
    "RandomForest_Iris": RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    ),
    "KNN_Iris": KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5)
    )
}

trusted_types = ["sklearn.tree._tree.Tree", "sklearn.neighbors._classification.KNeighborsClassifier"]

all_m = {}
for name, m in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_input(dataset, context="training")
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="weighted")),
            "recall": float(recall_score(y_test, pred, average="weighted")),
            "f1_score": float(f1_score(y_test, pred, average="weighted"))
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=m,
            artifact_path="model",
            registered_model_name=f"{name}_Model",
            skops_trusted_types=trusted_types
        )
        all_m[name] = metrics

os.makedirs("models", exist_ok=True)
joblib.dump(models["RandomForest_Iris"], "models/model.pkl")

os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(all_m, f, indent=4)

print("Entraînement terminé :", all_m)
