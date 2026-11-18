
import os
import pandas as pd
from sklearn.datasets import load_iris

print("\n========== PREPARING DATA ==========")

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

os.makedirs(DATA_DIR, exist_ok=True)

# Load Iris dataset
iris = load_iris(as_frame=True)

# Reference dataset
reference_data = iris.data.copy()

reference_data.to_csv(
    os.path.join(DATA_DIR, "reference_data.csv"),
    index=False
)

print("Reference dataset:", reference_data.shape)

# Simulated production dataset
current_data = reference_data.sample(
    n=150,
    replace=True,
    random_state=42
).copy()

# Add correct labels for supervised retraining
current_data["target"] = iris.target.loc[
    current_data.index
].values

# Introduce artificial drift
current_data["sepal length (cm)"] += 1.0
current_data["petal length (cm)"] += 0.8

current_data.to_csv(
    os.path.join(DATA_DIR, "current_data.csv"),
    index=False
)

print("Current dataset:", current_data.shape)
print("Target column included:", "target" in current_data.columns)

print("\nDataset preparation completed!")
