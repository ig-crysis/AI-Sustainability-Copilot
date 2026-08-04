import os
import re
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent.tools import (
    predict_footprint,
    get_live_carbon_intensity,
    compare_transport_scenarios,
    get_regional_baseline,
    TRANSPORT_EF,
    ENERGY_EF,
)

load_dotenv()

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI sustainability copilot. Your only job right now is to
extract structured facts from the user's message and call predict_footprint
with them — do not write any explanation or commentary, just call the tool.

CRITICAL RULES for calling predict_footprint:
- ONLY extract raw facts the user explicitly mentioned — NEVER compute,
  multiply, add, or average anything yourself. predict_footprint takes only
  atomic quantities (a count, an hours/day number, a category label) and
  does all arithmetic internally. If you catch yourself about to write a
  math expression as an argument value, stop — extract the raw fact instead.
- If the user did NOT mention food, set food_type="vegetables" and
  meals_with_this_food_per_week=0
- If the user did NOT mention flights, set flights_per_year=0 and
  avg_km_per_flight=0
- If the user did NOT mention a vehicle, set transport_type="bicycle" and
  km_per_day=0
- meals_with_this_food_per_week is a plain count 0-7 ("once a week"=1,
  "twice a week"=2, "every day"=7) — do not convert this to kg yourself.
  If food is mentioned but no frequency is stated (e.g. "I eat chicken"),
  assume 7 (daily) — that is the natural reading, not "not mentioned"
- If the user states an exact food quantity instead of a frequency (e.g.
  "200g of rice a day"), pass it directly as total_kg_food_per_day (in kg)
  and leave meals_with_this_food_per_week=0 — do not use both for the same
  food
- flights_per_year is a plain count of flights/trips — do not multiply by
  distance yourself; pass the distance separately as avg_km_per_flight
- Device hours (phone/laptop/desktop/tv_hours_per_day) and
  shower_frequency (daily/less_frequent/twice_daily/none) are passed as-is;
  default all device hours to 0 and shower_frequency to "daily" if not
  mentioned
- If the user states a specific total daily electricity usage directly
  (e.g. "use 15 kWh/day"), pass that number as total_kwh_per_day and leave
  all device-hour fields at 0 — do NOT also guess device hours in that case
- country: extract the user's country if mentioned (e.g. "india",
  "germany"); default "india" if not mentioned
- NEVER assume or hallucinate values the user did not provide

LIFESTYLE FEATURE RULES (extract if mentioned):
- Waste: infer waste_bag_size from context (small/medium/large/extra large)
  and waste_bags_per_week. Default: medium, 1 bag/week
- Clothing: new_clothes_per_month — if not mentioned, default 2
- Grocery: grocery_bill_monthly in USD equivalent — if not mentioned, default 200
- Energy efficiency: energy_efficient=true only if user explicitly says
  they use LED bulbs, efficient appliances, or similar

If the message describes or asks about ANY lifestyle activity (transport,
food, energy, travel, waste, shopping) or asks about their carbon footprint
at all — even vaguely, with no specifics — ALWAYS call predict_footprint,
using the defaults above for anything not mentioned.

If the message is entirely unrelated to lifestyle or carbon footprint (e.g.
small talk, general knowledge questions, anything off-topic), do NOT call
predict_footprint or any other tool. There is no other tool available to
you — do not invent one. Instead, write one short, friendly sentence
explaining you can only help with carbon footprint questions and asking
about their transport, food, or energy habits."""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert AI sustainability copilot. The user's carbon footprint has
already been computed for you (via a hybrid XGBoost + IPCC model) along with
supporting context — live grid carbon intensity, a transport-mode comparison,
and a regional per-capita baseline. Your only job is to write the natural-
language response using EXACTLY the numbers given below — do not recompute,
re-estimate, or contradict any of them.

Response requirements:
- State their monthly footprint in kg CO2 and, when a regional baseline is
  available, compare it to that baseline.
- Follow the THRESHOLD instruction given to you exactly — it tells you
  whether to congratulate, give gentle tips, give full suggestions, or use
  urgent language. Do not contradict it.
- When giving reduction suggestions, quantify them using the transport
  comparison data provided where relevant (e.g. "switching to train saves
  X kg CO2/month").
- Briefly note which inputs were assumed (defaults for anything the user
  didn't mention) vs explicitly stated, if it's relevant context.
- Be precise, empathetic, and solution-focused. Keep it concise — a few
  short paragraphs, not a report."""


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


def _build_llm(temperature: float = 0.2):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )


def _is_tool_use_failed(exc: BaseException) -> bool:
    body = getattr(exc, "body", None)
    return isinstance(body, dict) and (body.get("error") or {}).get("code") == "tool_use_failed"


_NATIVE_FUNCTION_CALL_RE = re.compile(r"<function=(\w+)>(\{.*?\})", re.DOTALL)


def _recover_native_function_call(exc: BaseException, tool_name: str):
    """Groq's tool_use_failed rejects the ENTIRE response outright based on
    format alone, even when llama-3.1-8b-instant's extracted content is
    correct. Confirmed by direct repro (see research/LIMITATIONS.md): a
    query that fails deterministically (same result across repeated calls
    at temperature=0, so a same-request retry cannot help) turned out to
    have a fully correct, well-formed extraction in `failed_generation` —
    the model just emitted it in Llama's native `<function=name>{...}` text
    format instead of Groq's expected structured tool-call schema. This
    recovers that case by parsing the JSON out of the rejected text instead
    of discarding a perfectly good answer."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict) or not _is_tool_use_failed(exc):
        return None
    failed_generation = (body.get("error") or {}).get("failed_generation", "")
    match = _NATIVE_FUNCTION_CALL_RE.search(failed_generation)
    if not match or match.group(1) != tool_name:
        return None
    try:
        return json.loads(match.group(2))
    except json.JSONDecodeError:
        return None


_OFF_TOPIC_DECLINE = (
    "I'm your carbon footprint copilot — I can only help with questions "
    "about your transport, food, energy, or lifestyle emissions. Tell me "
    "about your daily habits and I'll estimate your footprint!"
)


def _run_extraction(llm, messages, tool_name: str, max_retries: int = 1):
    """Returns (tool_args, declined_content). tool_args is None only if the
    model legitimately declined to call the tool (e.g. an off-topic
    message); declined_content then carries whatever text it wrote instead.

    On a Groq tool_use_failed error: first try to recover a correct
    extraction from the rejected native-format text (see
    _recover_native_function_call — no extra API call, and works even for
    deterministic per-query failures a retry can't fix). Only if that fails
    does this retry the call outright, which helps for the model's
    remaining genuinely stochastic malformed-generation flakiness (measured
    ~7% in evaluate_agent.py).

    If every attempt still fails with tool_use_failed (e.g. the model
    hallucinates an entirely unrelated tool name for an off-topic message —
    confirmed reproducible with a fictional "brute_search" tool on
    "What's the weather like today?"), this treats it as a decline rather
    than raising: there's nothing recoverable in that response, and letting
    Groq's raw error body escape to the end user is worse than a generic
    fallback message. Any other (non-tool_use_failed) exception still
    propagates immediately."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            ai_msg = llm.invoke(messages)
        except groq.BadRequestError as e:
            if not _is_tool_use_failed(e):
                raise
            recovered = _recover_native_function_call(e, tool_name)
            if recovered is not None:
                return recovered, ""
            if attempt < max_retries:
                last_exc = e
                continue
            return None, _OFF_TOPIC_DECLINE

        tool_call = next(
            (tc for tc in (ai_msg.tool_calls or []) if tc["name"] == tool_name),
            None,
        )
        if tool_call is not None:
            return tool_call["args"], ""
        return None, ai_msg.content if isinstance(ai_msg.content, str) else ""
    raise last_exc


def run_agent(user_message: str, chat_history: list = None) -> dict:
    """
    Run the agent in exactly two LLM calls: one to extract structured facts
    and call predict_footprint, one to synthesize the final response from
    the (already-known-correct) tool results and threshold.

    This replaces an earlier design that let the LLM freely choose among 4
    tools in a ReAct loop (~4 sequential LLM calls) and then, if the
    resulting text didn't match the actual threshold, re-ran the whole loop
    a second time based on a keyword-matching heuristic (up to ~8 calls
    worst case). Each call resends the full system prompt + tool schemas +
    conversation history, and this Groq API key is rate-limited to 6000
    tokens/minute — a single chat request could exhaust that budget within
    1-2 calls, triggering silent 429-retry-with-backoff waits of 20-35s+
    per call (confirmed via HTTP-level tracing; see research/LIMITATIONS.md).
    Fixing the threshold from the actual model output before generating any
    text also makes the old rerun heuristic unnecessary, not just slower.
    """
    if chat_history is None:
        chat_history = []

    # temperature=0 for extraction: this call must reliably emit a
    # structured tool call, not prose — determinism reduces the model's
    # chance of falling back to its native <function=...> text format
    # (Groq tool_use_failed). The synthesis call below writes free text,
    # where a little variation is fine, so it keeps the higher temperature.
    extraction_llm = _build_llm(temperature=0.0).bind_tools([predict_footprint])
    messages = (
        [SystemMessage(content=EXTRACTION_SYSTEM_PROMPT)]
        + chat_history
        + [HumanMessage(content=user_message)]
    )
    tool_args, declined_content = _run_extraction(
        extraction_llm, messages, "predict_footprint"
    )

    if tool_args is None:
        # Model declined to call the tool (e.g. an off-topic message) —
        # nothing to synthesize against, just return what it said.
        return {
            "output":    declined_content,
            "steps":     [],
            "threshold": "UNKNOWN",
            "actual_co2": None,
            "breakdown": None,
        }

    steps = [{"tool": "predict_footprint", "input": tool_args}]

    try:
        pred_result = predict_footprint.invoke(tool_args)
    except Exception:
        pred_result = {}
    actual_monthly_co2 = pred_result.get("predicted_monthly_co2_kg")

    # ── Deterministic enrichment — no LLM judgment needed for these, so no
    #    reason to spend a tool-choice round trip on each one ───────────────
    country     = tool_args.get("country", "india")
    km_per_day  = tool_args.get("km_per_day", 0.0)

    live_intensity = get_live_carbon_intensity.invoke({"country": country})
    steps.append({"tool": "get_live_carbon_intensity", "input": {"country": country}})

    transport_compare = compare_transport_scenarios.invoke({"km_per_day": km_per_day})
    steps.append({"tool": "compare_transport_scenarios",
                  "input": {"km_per_day": km_per_day, "days": 30}})

    regional_baseline = get_regional_baseline.invoke({"country": country})
    steps.append({"tool": "get_regional_baseline", "input": {"country": country}})

    threshold_instruction = (
        get_threshold_instruction(actual_monthly_co2)
        if actual_monthly_co2 is not None else ""
    )

    # ── Call 2 of 2: synthesize the final response from known-correct data ──
    context_block = f"""User's original message: {user_message}

COMPUTED RESULTS (use these exact numbers, do not recompute):
- Monthly footprint: {actual_monthly_co2} kg CO2/month
- Annual estimate: {pred_result.get('annual_estimate_kg')} kg CO2/year
- Category breakdown: {pred_result.get('breakdown')}
- Method: {pred_result.get('method')}
- Live grid carbon intensity: {live_intensity.get('carbon_intensity_gco2_kwh')} gCO2/kWh ({live_intensity.get('source')})
- Transport comparison (30 days at {km_per_day} km/day): best={transport_compare.get('best_option')}, worst={transport_compare.get('worst_option')}, max_saving_kg={transport_compare.get('max_saving_kg')}
- Regional baseline: {regional_baseline}

{threshold_instruction}"""

    synthesis_messages = (
        [SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT)]
        + chat_history
        + [HumanMessage(content=context_block)]
    )
    final_msg = _build_llm().invoke(synthesis_messages)
    final_response = final_msg.content if isinstance(final_msg.content, str) else ""

    return {
        "output":    final_response,
        "steps":     steps,
        "threshold": threshold_instruction.split(".")[0] if threshold_instruction else "UNKNOWN",
        "actual_co2": actual_monthly_co2,
        "breakdown": pred_result.get("breakdown"),
    }

# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Checking tools load correctly...")

    try:
        from agent.tools import booster, encoders, scaler
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
