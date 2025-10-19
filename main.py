from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Student ML Predictor")

# Setup templates & static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    hours_studied: float = Form(...),
    attendance: float = Form(...),
    assignments_done: int = Form(...),
    sleep_hours: float = Form(...)
):
    features = np.array([[hours_studied, attendance, assignments_done, sleep_hours]])
    prediction = model.predict(features)[0]
    result = "Pass ✅" if prediction == 1 else "Fail ❌"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
