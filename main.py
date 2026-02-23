from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load trained components
model = joblib.load("predict_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.get("/")
def home():
    return {"message": "Diabetes Prediction API"}


@app.post("/predict")
def predict(data: DiabetesInput):

    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # Transform (NOT fit_transform)
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    return {
        "predicted_Outcome": int(prediction),
        "meaning": "Diabetic" if prediction == 1 else "Not Diabetic"
    }

