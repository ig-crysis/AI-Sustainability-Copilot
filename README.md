# AI Sustainability Copilot

A chat-based carbon footprint estimator: a LangGraph agent (Groq-hosted LLM)
extracts lifestyle facts from natural language, calls a trained XGBoost
regression model for the footprint estimate, and enriches it with live grid
carbon-intensity data and country-level baselines.

## Architecture

```
frontend/  React + Vite chat UI (Vercel)
backend/   FastAPI (Render)
  main.py             — /chat, /predict, /health routes
  agent/carbon_agent.py — LangGraph ReAct agent (llama-3.1-8b-instant via Groq)
  agent/tools.py        — predict_footprint + 3 enrichment tools
  model/artifacts/      — trained XGBoost (best_model.pkl/.ubj) + RandomForest
  data/                 — raw survey dataset -> preprocessing -> train/test splits
```

**Prediction pipeline** (`agent/tools.py::predict_footprint`): the LLM
extracts only atomic facts from the user's message (a distance, a device's
daily usage hours, a meal frequency, a category label) — it does no
arithmetic. The tool itself:
1. Computes derived quantities (`kg_food_per_day`, `kwh_per_day`,
   `flight_km_total`) from those atomic facts.
2. Runs the XGBoost model (trained directly on real survey data) as the
   primary footprint estimate.
3. Corrects for real-time electricity grid carbon intensity (via
   Electricity Maps, with a static fallback) as an additive delta.
4. Falls back to an IPCC/Poore & Nemecek physical formula only when inputs
   fall outside the model's training distribution.

The IPCC/Poore & Nemecek formula is also used to produce the illustrative
per-category breakdown (transport/food/energy/flights/waste/clothing) shown
to the user — that breakdown is explanatory, not the calibrated total.

## Setup

### Backend
```bash
cd backend
python -m venv venv && source venv/Scripts/activate   # or venv/bin/activate on Linux/Mac
pip install -r requirements.txt
# .env with GROQ_API_KEY (required) and CO2SIGNAL_API_KEY (optional, live grid data)
uvicorn main:app --reload
```

To reproduce the training pipeline from scratch (not required to run the
app — trained artifacts are already committed), see the reproducibility
note in `backend/data/README.md`: the raw dataset isn't committed and needs
to be sourced manually.

### Frontend
```bash
cd frontend
npm install
# .env with VITE_API_URL pointing at the backend
npm run dev
```

## Model performance

See `backend/evaluate_ablation.py` for the full ablation (run it yourself —
numbers below are from the held-out test set, 2,000 rows, 80/20 split,
`random_state=42`):

| Model | MAE (kg CO₂/month) | R² |
|---|---|---|
| Mean baseline | 65.16 | 0.00 |
| IPCC formula only (superseded primary) | 454.73 | -56.02 |
| RandomForest | 27.47 | 0.80 |
| **XGBoost (deployed primary)** | **25.30** | **0.83** (95% CI [0.81, 0.84]) |

The IPCC-only row is intentionally kept in the table: an earlier production
version used IPCC as the primary predictor with XGBoost as a bounded ±15%
adjustment, which scored R²=-73.8 on this same test set. See
`research/LIMITATIONS.md` for that incident's full writeup.

**Agent (LLM extraction) evaluation** — `backend/evaluate_agent.py`, 12
hand-authored natural-language test cases against the live Groq API
(`llama-3.1-8b-instant`), re-run before citing since results can drift with
model updates:

| Metric | Result |
|---|---|
| Tool-call errors (malformed function calls) | 0/12 |
| Fully correct (all extracted fields match ground truth) | 10/12 (83.3%) |
| Mean per-field extraction accuracy | 98.7% |

An earlier tool design asked the LLM to pre-compute derived quantities
itself (e.g. sum device-hours into a single `kwh_per_day`); that caused
**11/12 (92%) of queries to fail outright** with a Groq `tool_use_failed`
error, because the model would sometimes emit an unevaluated arithmetic
expression instead of a number. Fixed by moving all arithmetic out of the
LLM's responsibility into the tool itself — see `research/LIMITATIONS.md`.

## Citations

- Transport/flight emission factors: IPCC AR6 Working Group III.
- Food emission factors: Poore, J. & Nemecek, T. (2018). "Reducing food's
  environmental impacts through producers and consumers." *Science*.
- Regional per-capita baselines: Our World in Data (OWID), based on the
  Global Carbon Project.
- Live grid carbon intensity: Electricity Maps API.
- Primary training dataset: `backend/data/raw/Carbon_Emission.csv` —
  **citation TODO**, see `backend/data/README.md`.

## Known limitations

See `research/LIMITATIONS.md`.
