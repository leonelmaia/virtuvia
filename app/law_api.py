# api.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from law_agent import (
    classify_case_gpt,
    analyze_risk,
    generate_mock_report,
    extract_keyword,
)
from utils import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()
openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


class ClassifyRequest(BaseModel):
    text: str


class RiskRequest(BaseModel):
    case_type: str
    value: float


class ReportRequest(BaseModel):
    area: str


class lawRequest(BaseModel):
    text: str


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/classify")
def classify_case(req: ClassifyRequest):
    try:
        case_type = classify_case_gpt(req.text, openai)
        return {"category": case_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/law")
def extract_law_subject(req: lawRequest):
    try:
        keyword = extract_keyword(req.text, openai)
        return {"keyword": keyword}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk")
def risk_analysis(req: RiskRequest):
    try:
        result = analyze_risk(req.case_type.title(), req.value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report")
def generate_report(req: ReportRequest):
    try:
        result = generate_mock_report(req.area.title())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
