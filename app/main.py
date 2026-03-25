"""
Transliteration Engine — Demo API
===================================
Lightweight FastAPI server for live transliteration.
Demonstrates the system as a mobile keyboard backend would use it.

Usage:
    uvicorn app.main:app --reload
    # Open http://localhost:8000 in your browser
"""

import os
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import TransliterationSystem
from src.config import DATA_DIR

# ==========================================
# APP INITIALIZATION
# ==========================================
app = FastAPI(
    title="Transliteration Engine API",
    description="Real-time Roman to Devanagari transliteration",
    version="1.0.0",
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Load the transliteration system on startup
system = None


@app.on_event("startup")
async def load_model():
    global system
    print("Loading Transliteration System...")
    system = TransliterationSystem(DATA_DIR)
    print("System ready!")


# ==========================================
# API MODELS
# ==========================================
class TransliterateRequest(BaseModel):
    text: str


class TransliterateResponse(BaseModel):
    input: str
    result: str
    latency_ms: float


# ==========================================
# ROUTES
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the demo web interface."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/transliterate", response_model=TransliterateResponse)
async def transliterate(req: TransliterateRequest):
    """
    Transliterate Roman text to Devanagari.

    Request body:
        {"text": "namaste doston"}

    Response:
        {"input": "namaste doston", "result": "नमस्ते दोस्तों", "latency_ms": 12.3}
    """
    start = time.perf_counter()
    result = system.transliterate(req.text)
    latency = (time.perf_counter() - start) * 1000

    return TransliterateResponse(
        input=req.text,
        result=result,
        latency_ms=round(latency, 2),
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": system is not None}
