
import os
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model.pkl"
)

app = FastAPI(
    title="Iris Classification API",
    description="MLOps model serving API",
    version="1.0.0"
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Run the training pipeline first."
    )

model = joblib.load(MODEL_PATH)

print("Iris model loaded successfully!")


# ==========================================
# INPUT VALIDATION
# ==========================================

class IrisInput(BaseModel):

    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four Iris flower measurements"
    )


# ==========================================
# HEALTH CHECK API
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }


# ==========================================
# MODEL PREDICTION API
# ==========================================

@app.post("/predict")
def predict(input_data: IrisInput):

    try:

        features = np.array(
            input_data.features
        ).reshape(1, -1)

        prediction = model.predict(features)

        predicted_class = int(prediction[0])

        class_names = [
            "setosa",
            "versicolor",
            "virginica"
        ]

        return {
            "model_version": "1.0.0",
            "prediction": predicted_class,
            "flower_name": class_names[predicted_class]
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# ROOT API
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to Iris Classification API",
        "docs": "/docs",
        "health": "/health"
    }
