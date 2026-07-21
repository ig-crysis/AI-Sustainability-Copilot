import os
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from agent.tools import (
    predict_footprint,
    get_live_carbon_intensity,
    compare_transport_scenarios,
    get_regional_baseline,
    TRANSPORT_EF,
    ENERGY_EF,
)

load_dotenv()

TOOLS = [
    predict_footprint,
    get_live_carbon_intensity,
    compare_transport_scenarios,
    get_regional_baseline,
]

SYSTEM_PROMPT = """You are an expert AI sustainability copilot. Your role is to:
1. Accurately estimate carbon footprints using the predict_footprint tool
2. Enrich predictions with live carbon intensity data via get_live_carbon_intensity
3. Compare transport alternatives using compare_transport_scenarios
4. Provide regional context using get_regional_baseline
5. Always give specific, quantified, actionable suggestions

CRITICAL RULES for calling predict_footprint:
- ONLY include data the user explicitly mentioned
- If the user did NOT mention food, set food_type="vegetables" and kg_food_per_day=0.0
- If the user did NOT mention flights, set flights_per_year=0 and flight_km_total=0
- If the user did NOT mention a vehicle, set transport_type="bicycle" and km_per_day=0
- NEVER assume or hallucinate values the user did not provide

FOOD QUANTITY RULES:
- "eats X once a week" = 0.15kg × 1/7 = 0.021 kg/day
- "eats X twice a week" = 0.15kg × 2/7 = 0.043 kg/day
- "eats X thrice a week" = 0.15kg × 3/7 = 0.064 kg/day
- "eats X every day" = 0.15 kg/day
- Standard meal portion = 150g = 0.15 kg

LIFESTYLE FEATURE RULES (extract if mentioned):
- Waste: infer waste_bag_size from context (small/medium/large/extra large)
  and waste_bags_per_week. Default: medium, 1 bag/week
- Clothing: new_clothes_per_month — if not mentioned, default 2
- Grocery: grocery_bill_monthly in USD equivalent — if not mentioned, default 200
- Energy efficiency: energy_efficient=true only if user explicitly says
  they use LED bulbs, efficient appliances, or similar

DEVICE ENERGY RULES:
- phone ~0.005 kWh/hr, laptop ~0.05 kWh/hr, desktop ~0.1 kWh/hr, TV ~0.1 kWh/hr
- shower: daily=0.9 kWh, less frequent=0.45 kWh, twice daily=1.8 kWh
- Add 2.0 kWh base household consumption

Workflow for every query:
- ALWAYS call predict_footprint first to get the ML model estimate
- Then call relevant enrichment tools based on user context
- Synthesize all tool outputs into a clear structured response
- Always quantify savings: "switching to train saves X kg CO2/month"
- Compare user footprint to regional average when country is known
- Rate footprint: Low/Moderate/High/Critical (global avg = 375 kg/month)
- Clearly state which data points were assumed vs explicitly provided

Be precise, empathetic, and solution-focused."""


def get_threshold_instruction(monthly_co2: float) -> str:
    """Returns threshold-based instruction to append to user message."""
    if monthly_co2 < 200:
        return (
            "THRESHOLD: LOW. The user's footprint is below 200 kg CO2/month. "
            "Do NOT provide reduction suggestions. Instead, congratulate them, "
            "tell them they are below the global average of 375 kg/month, and "
            "encourage them to maintain their lifestyle."
        )
    elif monthly_co2 < 375:
        return (
            "THRESHOLD: MODERATE. The user's footprint is between 200-375 kg CO2/month. "
            "Provide 1-2 gentle, easy-to-implement suggestions focused only on "
            "their biggest emission category. Keep it encouraging, not alarming."
        )
    elif monthly_co2 < 600:
        return (
            "THRESHOLD: HIGH. The user's footprint is between 375-600 kg CO2/month, "
            "above the global average. Provide full actionable suggestions across "
            "all emission categories with specific CO2 savings quantified."
        )
    else:
        return (
            "THRESHOLD: CRITICAL. The user's footprint exceeds 600 kg CO2/month, "
            "which is well above the global average of 375 kg/month. Use urgent but "
            "empathetic language. Prioritize the top 3 highest-impact changes with "
            "exact CO2 savings. Compare to global and regional averages."
        )


def build_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )
    return create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def run_agent(user_message: str, chat_history: list = None) -> dict:
    """Run the agent with threshold-aware suggestions."""
    if chat_history is None:
        chat_history = []

    # ── Step 1: Run agent WITHOUT threshold instruction first ────────────────
    agent    = build_agent()
    messages = chat_history + [HumanMessage(content=user_message)]
    result   = agent.invoke({"messages": messages})

    all_messages   = result["messages"]
    final_response = ""
    steps          = []
    actual_monthly_co2 = None

    for msg in all_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "tool":  tc["name"],
                    "input": tc["args"],
                })
        if msg.__class__.__name__ == "AIMessage":
            if isinstance(msg.content, str) and msg.content.strip():
                final_response = msg.content

    # ── Step 2: Extract actual CO2 from predict_footprint tool call ──────────
    for step in steps:
        if step["tool"] == "predict_footprint":
            try:
                from agent.tools import predict_footprint
                import xgboost as xgb
                pred_result = predict_footprint.invoke(step["input"])
                actual_monthly_co2 = pred_result.get("predicted_monthly_co2_kg", None)
            except Exception:
                pass
            break

    # ── Step 3: Determine threshold from ACTUAL model output ─────────────────
    threshold_instruction = ""
    if actual_monthly_co2 is not None:
        threshold_instruction = get_threshold_instruction(actual_monthly_co2)

        # ── Step 4: If threshold says LOW but response gives suggestions,
        #    re-run agent with correct threshold injected ─────────────────────
        threshold_level = threshold_instruction.split(".")[0]
        needs_rerun = False

        if threshold_level == "THRESHOLD: LOW" and any(
            phrase in final_response.lower()
            for phrase in ["consider", "reduce", "switch", "suggestion", "recommend"]
        ):
            needs_rerun = True
        elif threshold_level == "THRESHOLD: CRITICAL" and "congratulat" in final_response.lower():
            needs_rerun = True

        if needs_rerun:
            augmented = f"{user_message}\n\n[SYSTEM NOTE: {threshold_instruction}]"
            messages2 = chat_history + [HumanMessage(content=augmented)]
            result2   = build_agent().invoke({"messages": messages2})
            for msg in result2["messages"]:
                if msg.__class__.__name__ == "AIMessage":
                    if isinstance(msg.content, str) and msg.content.strip():
                        final_response = msg.content

    return {
        "output":    final_response,
        "steps":     steps,
        "threshold": threshold_instruction.split(".")[0] if threshold_instruction else "UNKNOWN",
        "actual_co2": actual_monthly_co2,
    }

# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Checking tools load correctly...")

    try:
        from agent.tools import model, encoders, scaler
        print("[OK] XGBoost model loaded")
        print("[OK] Encoders loaded")
        print("[OK] Scaler loaded")
    except Exception as e:
        print(f"[ERROR] Tool load failed: {e}")
        exit(1)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_key_here":
        print("\n[WARN] GROQ_API_KEY not set in .env")
    else:
        print("[OK] Groq API key found\n")

        # Test LOW threshold
        print("TEST 1 — Low footprint query:")
        print("=" * 60)
        low_query = "I walk to work 1km daily and use 1 kWh/day at home in India."
        result = run_agent(low_query)
        print("Threshold:", result["threshold"])
        print("Response:", result["output"][:300])

        print("\nTEST 2 — Critical footprint query:")
        print("=" * 60)
        critical_query = (
            "I drive a petrol car 80km daily, eat beef every day (500g), "
            "use 40 kWh/day, and fly every month (total 20000km/year) in the US."
        )
        result = run_agent(critical_query)
        print("Threshold:", result["threshold"])
        print("Response:", result["output"][:300])