import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from agent.carbon_agent import run_agent

load_dotenv()

app = FastAPI(
    title="AI Sustainability Copilot API",
    description="Agentic AI pipeline for carbon footprint analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://ai-sustainability-copilot.vercel.app"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    location:   Optional[dict] = None

class ToolStep(BaseModel):
    tool: str
    input: dict

class ChatResponse(BaseModel):
    response: str
    tools_called: list[ToolStep]
    session_id: str
    threshold:    str = "UNKNOWN"
    actual_co2:   Optional[float] = None

class FootprintRequest(BaseModel):
    transport_type: str
    km_per_day: float
    food_type: str
    meals_with_this_food_per_week: float
    energy_source: str
    phone_hours_per_day: float = 0.0
    laptop_hours_per_day: float = 0.0
    desktop_hours_per_day: float = 0.0
    tv_hours_per_day: float = 0.0
    shower_frequency: str = "daily"
    flights_per_year: int = 0
    avg_km_per_flight: float = 0.0

class FootprintResponse(BaseModel):
    predicted_monthly_co2_kg: float
    annual_estimate_kg: float
    breakdown: dict
    rating: str
    agent_analysis: str
    tools_called: list[ToolStep]


# ── In-memory session store (replace with Redis in production) ───────────────
session_store: dict[str, list] = {}


def get_rating(monthly_kg: float) -> str:
    if monthly_kg < 200:   return "Low"
    if monthly_kg < 375:   return "Moderate"
    if monthly_kg < 600:   return "High"
    return "Critical"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "AI Sustainability Copilot",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/chat", "/predict", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost R²=0.83 (synthetic data, see research/LIMITATIONS.md)", "llm": "llama-3.1-8b-instant"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Send a natural language message,
    get back an agent response with tool call trace.
    """
    try:
        if request.location:
            print(f"[Location] {request.location.get('city')}, {request.location.get('country')}")
        history = session_store.get(request.session_id, [])
        result  = run_agent(request.message, chat_history=history)

        # Update session history
        from langchain_core.messages import HumanMessage, AIMessage
        history.append(HumanMessage(content=request.message))
        history.append(AIMessage(content=result["output"]))
        session_store[request.session_id] = history[-20:]  # keep last 10 turns

        return ChatResponse(
            response=result["output"],
            tools_called=[ToolStep(**s) for s in result["steps"]],
            session_id=request.session_id,
            threshold=result.get("threshold", "UNKNOWN"),
            actual_co2=result.get("actual_co2"),
            
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] /chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong processing your message. Please try rephrasing it.",
        )


@app.post("/predict", response_model=FootprintResponse)
async def predict(request: FootprintRequest):
    """
    Structured prediction endpoint. Takes explicit activity parameters,
    runs the full agentic pipeline, returns structured + explained result.
    """
    try:
        # Call the tool directly with the caller's structured, already-atomic
        # facts — no LLM round-trip needed (or wanted) for the numeric result.
        from agent.tools import predict_footprint
        pred = predict_footprint.invoke(request.model_dump())
        monthly_co2 = pred.get("predicted_monthly_co2_kg", 0.0)
        breakdown   = pred.get("breakdown", {})

        # Still run the agent for a natural-language analysis/commentary
        # (regional comparison, transport alternatives, etc.).
        query = (
            f"I use a {request.transport_type.replace('_', ' ')} for "
            f"{request.km_per_day}km daily, eat {request.food_type} "
            f"{request.meals_with_this_food_per_week} times a week, use "
            f"{request.energy_source.replace('_', ' ')} electricity, and take "
            f"{request.flights_per_year} flights per year averaging "
            f"{request.avg_km_per_flight}km each. "
            f"Analyze my carbon footprint in detail."
        )
        result = run_agent(query)

        return FootprintResponse(
            predicted_monthly_co2_kg=monthly_co2,
            annual_estimate_kg=round(monthly_co2 * 12, 2),
            breakdown=breakdown,
            rating=get_rating(monthly_co2),
            agent_analysis=result["output"],
            tools_called=[ToolStep(**s) for s in result["steps"]],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] /predict failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong processing your request. Please check your inputs and try again.",
        )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    session_store.pop(session_id, None)
    return {"message": f"Session {session_id} cleared"}