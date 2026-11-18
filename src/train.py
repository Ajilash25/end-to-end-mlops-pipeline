
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "model.pkl"
)

MLFLOW_DB_PATH = os.path.join(
    PROJECT_ROOT, "mlflow.db"
)

print("\n========== INITIAL MODEL TRAINING ==========")


# ==========================================
# CONFIGURE MLFLOW
# ==========================================

mlflow.set_tracking_uri(
    "sqlite:///" + MLFLOW_DB_PATH.replace("\\", "/")
)

mlflow.set_experiment("Iris-Classification")

print("MLflow tracking database:", MLFLOW_DB_PATH)


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading Iris dataset...")

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:", accuracy)


# ==========================================
# MLFLOW EXPERIMENT TRACKING
# ==========================================

with mlflow.start_run(run_name="Initial-Training"):

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris-model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print("\nModel logged successfully in MLflow!")


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)


# ==========================================
# COMPLETION
# ==========================================

print("\n========== TRAINING COMPLETED ==========")
print("Initial model accuracy:", accuracy)
