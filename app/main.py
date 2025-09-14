from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
import os
# import requests


universities = {
    'FAST': {'difficulty': 7, 'fields': ['CS']},
    'NUST': {'difficulty': 8, 'fields': ['CS', 'Engineering', 'Medicine']},
    'UET': {'difficulty': 6, 'fields': ['Engineering']},
    'COMSAT': {'difficulty': 5, 'fields': ['CS', 'Business']},
    'GIKI': {'difficulty': 9, 'fields': ['Engineering', 'CS']}
}
def generate_time_pct():
        """
        Returns a fraction of allotted time: N(1.0, 0.1) clipped [0.4,1.0].
        """
        return float(np.clip(np.random.normal(1.0, 0.1), 0.4, 1.0))

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # absolute path of main.py


# Stage 1 pipeline
admission_model = joblib.load("stage1_pipeline.pkl")

# Stage 2 pipeline
university_model = joblib.load("stage2_pipeline.pkl")
# admission_model = joblib.load(os.path.join(BASE_DIR, "stage1_pipeline.pkl"))
# university_model = joblib.load(os.path.join(BASE_DIR, "stage2_pipeline.pkl"))

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
    request: Request,
    physics_marks: float = Form(...),
    math_marks: float = Form(...),
    english_marks: float = Form(...),
    number_of_attempts: int = Form(...),
    preferred_field: str = Form(...),
    choice1: str = Form(...),
    choice2: str = Form(...),
    choice3: str = Form(...),

):
    # Prepare initial DataFrame
    df = pd.DataFrame([{
        "physics_marks": physics_marks,
        "math_marks": math_marks,
        "english_marks": english_marks,
        "number_of_attempts": number_of_attempts,
        "preferred_field": preferred_field,
        "choice1": choice1,
        "choice2": choice2,
        "choice3": choice3,

    }])

    # Feature Engineering
    df['average_marks'] = (physics_marks + math_marks + english_marks) / 3
    df['phy_norm'] = physics_marks / 100
    df['math_norm'] = math_marks / 100
    df['eng_norm'] = english_marks / 100

    df['eff_phy'] = generate_time_pct() * df['phy_norm']
    df['eff_math'] = generate_time_pct() * df['math_norm']
    df['eff_eng'] = generate_time_pct() * df['eng_norm']

    df['choice1_diff'] = universities[choice1]['difficulty']
    df['choice2_diff'] = universities[choice2]['difficulty']
    df['choice3_diff'] = universities[choice3]['difficulty']
    df['attempt_x_diff'] = number_of_attempts * (math_marks - physics_marks)
    for uni in universities:
        diff = 60 + 5 * list(universities.keys()).index(uni)
        df[f'perf_gap_{uni}'] = df['average_marks'] - diff

    # Feature lists
    numeric_feats = [
        'average_marks', 'phy_norm', 'math_norm', 'eng_norm',
        'number_of_attempts', 'eff_phy', 'eff_math', 'eff_eng','attempt_x_diff',
        'choice1_diff', 'choice2_diff', 'choice3_diff'
    ] + [f'perf_gap_{u}' for u in universities]

    categorical_feats = ['preferred_field', 'choice1', 'choice2', 'choice3']

    # Make predictions
    admitted = admission_model.predict(df[numeric_feats + categorical_feats])[0]
    result = {"admitted": bool(admitted)}

    if result["admitted"]:
        result["recommended_university"] = university_model.predict(df[numeric_feats + categorical_feats])[0]
    else:
        result["recommended_university"] = "Need Improvement" # type: ignore

    return templates.TemplateResponse("result.html", {
        "request": request,
        "admitted": result["admitted"],
        "university": result["recommended_university"]
    })
