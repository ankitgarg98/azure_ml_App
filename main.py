# main.py
from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Simple ML API", description="Predict student pass/fail", version="1.0")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class StudentInput(BaseModel):
    hours_studied: float
    attendance: float
    assignments_done: int
    sleep_hours: float

@app.get("/")
def home():
    return {"message": "Welcome to the Student Performance Predictor API!"}

@app.post("/predict")
def predict(data: StudentInput):
    features = np.array([[data.hours_studied, data.attendance, data.assignments_done, data.sleep_hours]])
    prediction = model.predict(features)[0]
    result = "Pass" if prediction == 1 else "Fail"
    return {"prediction": result}
