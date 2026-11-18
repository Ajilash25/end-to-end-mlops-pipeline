
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "current_data.csv"
)

RESULT_PATH = os.path.join(
    PROJECT_ROOT, "reports", "drift_result.json"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "model.pkl"
)


# ==========================================
# RETRAINING PIPELINE
# ==========================================

def retrain_model():

    print("\n========== AUTOMATIC RETRAINING PIPELINE ==========")

    # Step 1: Check drift result
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError(
            "Drift result not found. Run drift_monitor.py first."
        )

    with open(RESULT_PATH, "r") as file:
        drift_result = json.load(file)

    if not drift_result["drift_detected"]:

        print("\nNo data drift detected.")
        print("Retraining is not required.")

        return

    print("\nData drift detected!")
    print("Drift share:", drift_result["drift_share"])
    print("Starting model retraining...")

    # Step 2: Load current dataset
    data = pd.read_csv(DATA_PATH)

    if "target" not in data.columns:
        raise ValueError(
            "Target column is missing from current_data.csv"
        )

    X = data.drop(columns=["target"])
    y = data["target"]

    print("\nCurrent dataset loaded:", data.shape)
    print("Features:", X.shape)
    print("Target:", y.shape)

    # Step 3: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 4: Load existing model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. Train the initial model first."
        )

    old_model = joblib.load(MODEL_PATH)

    old_predictions = old_model.predict(X_test)

    old_accuracy = accuracy_score(
        y_test,
        old_predictions
    )

    print("\n========== OLD MODEL ==========")
    print("Existing model accuracy:", old_accuracy)

    # Step 5: Train new model
    print("\nTraining new Random Forest model...")

    new_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    new_model.fit(X_train, y_train)

    new_predictions = new_model.predict(X_test)

    new_accuracy = accuracy_score(
        y_test,
        new_predictions
    )

    print("\n========== NEW MODEL ==========")
    print("New model accuracy:", new_accuracy)

    # Step 6: Log experiment in MLflow
    mlflow.set_tracking_uri(
        "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
    )

    mlflow.set_experiment("Iris-Classification")

    with mlflow.start_run(
        run_name="Automatic-Retraining"
    ):

        mlflow.log_param(
            "algorithm",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            100
        )

        mlflow.log_param(
            "retraining_reason",
            "Data Drift"
        )

        mlflow.log_metric(
            "old_accuracy",
            old_accuracy
        )

        mlflow.log_metric(
            "new_accuracy",
            new_accuracy
        )

        mlflow.log_metric(
            "accuracy_improvement",
            new_accuracy - old_accuracy
        )

        mlflow.sklearn.log_model(
            sk_model=new_model,
            name="retrained-iris-model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\nNew model logged in MLflow!")

    # Step 7: Model validation and promotion
    print("\n========== MODEL VALIDATION ==========")

    if new_accuracy >= old_accuracy:

        joblib.dump(new_model, MODEL_PATH)

        print("New model accepted!")
        print("Production model updated.")

    else:

        print("New model rejected!")
        print("Existing model retained.")

    # Step 8: Final result
    print("\n========== RETRAINING COMPLETED ==========")

    print("Old accuracy:", old_accuracy)
    print("New accuracy:", new_accuracy)

    print(
        "Accuracy improvement:",
        new_accuracy - old_accuracy
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    retrain_model()
