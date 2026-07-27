"""
Agent evaluation harness — natural-language extraction accuracy.

The regression model has a rigorous offline evaluation (evaluate_ablation.py),
but nothing previously checked whether the LLM agent (agent/carbon_agent.py,
llama-3.1-8b-instant via Groq) correctly extracts structured facts from
free-text queries before handing them to predict_footprint. This script is
that check.

predict_footprint's arguments are all atomic facts (a count, an hours/day
number, a category label) — the tool does its own arithmetic internally
(see agent/tools.py). An earlier version of the tool asked the LLM to
pre-compute derived quantities itself (e.g. summing device kWh rates into a
single kwh_per_day); that caused ~92% of test queries to fail outright with
a Groq tool_use_failed error, because the model would sometimes emit an
unevaluated expression like "0.05*4+0.9" instead of a number. This is now a
regression check for that failure mode as much as an accuracy check.

Scoring:
  - Categorical fields (transport_type, food_type, energy_source,
    shower_frequency): exact match (case-insensitive).
  - energy_efficient: exact boolean match.
  - Numeric fields: match if within a relative tolerance of the ground
    truth (default 15%), or an absolute tolerance when the ground truth is
    ~0 (relative tolerance is meaningless near zero).

IMPORTANT: this hits the live Groq API — non-deterministic (temperature=0.2,
not 0), costs real API calls, and results can drift if the model version
changes. Every run prints the model name and should be re-run (not assumed
stable) before citing numbers in a paper. Requires GROQ_API_KEY to be set.
"""

import sys
import time
from statistics import mean

sys.path.insert(0, ".")
from agent.carbon_agent import run_agent  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_NAME = "llama-3.1-8b-instant"  # must match agent/carbon_agent.py::build_agent

CATEGORICAL_FIELDS = ["transport_type", "food_type", "energy_source", "shower_frequency"]
NUMERIC_FIELDS = [
    "km_per_day", "meals_with_this_food_per_week", "total_kg_food_per_day",
    "phone_hours_per_day", "laptop_hours_per_day",
    "desktop_hours_per_day", "tv_hours_per_day", "total_kwh_per_day",
    "flights_per_year", "avg_km_per_flight",
]
ABS_TOL = {
    "km_per_day": 2.0, "meals_with_this_food_per_week": 0.5,
    "total_kg_food_per_day": 0.05,
    "phone_hours_per_day": 0.5, "laptop_hours_per_day": 0.5,
    "desktop_hours_per_day": 0.5, "tv_hours_per_day": 0.5,
    "total_kwh_per_day": 1.0,
    "flights_per_year": 0.4, "avg_km_per_flight": 250.0,
}
REL_TOL = 0.15

# This Groq API key is capped at 6000 tokens/minute (see research/LIMITATIONS.md).
# Each case makes 2 LLM calls (~2000-2200 tokens combined); firing all cases
# back-to-back blows the budget and Groq degrades into malformed
# tool_use_failed 400s instead of clean 429s, which looks like an accuracy
# regression but is actually a burst-load artifact. Pace requests to stay
# under the budget so this harness measures steady-state accuracy.
SLEEP_BETWEEN_CASES_SECONDS = 20

# ── Test cases ────────────────────────────────────────────────────────────────
CASES = [
    dict(
        id="explicit_all_fields",
        query=("I drive a petrol car for 40km every day. I eat chicken every "
               "day. My electricity comes from the grid in the US, and I use "
               "my laptop 4 hours a day plus a daily shower. I don't fly."),
        expected=dict(transport_type="car_petrol", km_per_day=40.0,
                      food_type="chicken", meals_with_this_food_per_week=7,
                      energy_source="grid_us", laptop_hours_per_day=4.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="default_no_transport_mentioned",
        query=("I eat dairy every day. My home runs on natural gas heating, "
               "I use my phone 5 hours a day and shower daily. I never fly."),
        expected=dict(transport_type="bicycle", km_per_day=0.0,
                      food_type="dairy", meals_with_this_food_per_week=7,
                      energy_source="natural_gas", phone_hours_per_day=5.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="default_no_food_mentioned",
        query=("I drive a diesel car 25km per day to work. I use hydro power "
               "at home, run my TV for 3 hours a day, shower less frequently "
               "than daily. I flew once this year, a 4000km round trip."),
        expected=dict(transport_type="car_diesel", km_per_day=25.0,
                      food_type="vegetables", meals_with_this_food_per_week=0,
                      energy_source="hydro", tv_hours_per_day=3.0,
                      shower_frequency="less_frequent",
                      flights_per_year=1, avg_km_per_flight=4000.0, energy_efficient=False),
    ),
    dict(
        id="default_no_flights_mentioned",
        query=("I take the bus 15km every day. I eat fish every day. My "
               "electricity is from a coal plant, and I use my desktop for "
               "6 hours a day with a daily shower."),
        expected=dict(transport_type="bus", km_per_day=15.0,
                      food_type="fish", meals_with_this_food_per_week=7,
                      energy_source="coal", desktop_hours_per_day=6.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="food_once_a_week",
        query=("I ride an electric car 35km a day. I eat beef once a week, "
               "vegetables otherwise. Solar panels power my home, laptop 2 "
               "hours a day, daily shower. No flights."),
        expected=dict(transport_type="car_ev", km_per_day=35.0,
                      food_type="beef", meals_with_this_food_per_week=1,
                      energy_source="solar", laptop_hours_per_day=2.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="food_twice_a_week",
        query=("I take the train 60km daily for commuting. I eat lamb twice "
               "a week. Wind power at home, phone 3 hours and TV 2 hours a "
               "day, shower daily. No flights this year."),
        expected=dict(transport_type="train", km_per_day=60.0,
                      food_type="lamb", meals_with_this_food_per_week=2,
                      energy_source="wind", phone_hours_per_day=3.0, tv_hours_per_day=2.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="food_thrice_a_week",
        query=("I ride a motorcycle 20km every day. I eat pork three times "
               "a week. Nuclear power at home, desktop 4 hours a day, shower "
               "twice a day. I flew twice this year, 3000km each trip."),
        expected=dict(transport_type="motorcycle", km_per_day=20.0,
                      food_type="pork", meals_with_this_food_per_week=3,
                      energy_source="nuclear", desktop_hours_per_day=4.0,
                      shower_frequency="twice_daily",
                      flights_per_year=2, avg_km_per_flight=3000.0, energy_efficient=False),
    ),
    dict(
        id="explicit_walking",
        query=("I walk 5km every day, no vehicle. I eat eggs every day. Oil "
               "heating at home, phone 6 hours a day, shower daily. No flights."),
        expected=dict(transport_type="walking", km_per_day=5.0,
                      food_type="eggs", meals_with_this_food_per_week=7,
                      energy_source="oil", phone_hours_per_day=6.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="explicit_bicycle_with_distance",
        query=("I cycle 12km every day to work. I eat rice every day. Grid "
               "electricity in Germany, laptop 5 hours a day, shower daily. "
               "No flights."),
        expected=dict(transport_type="bicycle", km_per_day=12.0,
                      food_type="rice", meals_with_this_food_per_week=7,
                      energy_source="grid_eu", laptop_hours_per_day=5.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="vegan_multi_device",
        query=("I don't own a car and don't commute. I'm vegan and eat "
               "legumes every day. Solar power at home, I use my phone 4 "
               "hours, laptop 3 hours, and TV 2 hours a day, shower less "
               "frequently. No air travel."),
        expected=dict(transport_type="bicycle", km_per_day=0.0,
                      food_type="legumes", meals_with_this_food_per_week=7,
                      energy_source="solar", phone_hours_per_day=4.0,
                      laptop_hours_per_day=3.0, tv_hours_per_day=2.0,
                      shower_frequency="less_frequent",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="energy_efficient_explicit",
        query=("I drive a petrol car 18km every day. I eat fruits every day. "
               "Grid electricity in India, but I use LED bulbs and "
               "energy-efficient appliances throughout my home. Laptop 3 "
               "hours a day, shower daily. No flights."),
        expected=dict(transport_type="car_petrol", km_per_day=18.0,
                      food_type="fruits", meals_with_this_food_per_week=7,
                      energy_source="grid_india", laptop_hours_per_day=3.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=True),
    ),
    dict(
        # Regression test for a live production bug: user states a total
        # daily kWh directly, and mentions food with no explicit frequency.
        # Before the fix, the direct kWh statement was silently dropped
        # (no field existed to carry it), and ambiguous food defaulted to
        # kg=0 instead of assuming daily. See research/LIMITATIONS.md.
        id="direct_kwh_stated_and_ambiguous_food_frequency",
        query=("I drive a petrol car 30km daily, eat chicken, use 15kWh/day "
               "in India. What's my footprint?"),
        expected=dict(transport_type="car_petrol", km_per_day=30.0,
                      food_type="chicken", meals_with_this_food_per_week=7,
                      energy_source="grid_india", total_kwh_per_day=15.0,
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        # Regression test: user states an exact food quantity instead of a
        # frequency — should route to total_kg_food_per_day, not be forced
        # through the meals-per-week model.
        id="direct_food_quantity_stated",
        query=("I take the bus 20km a day. I eat 300g of rice daily. Wind "
               "power at home, laptop 3 hours a day, shower daily. No flights."),
        expected=dict(transport_type="bus", km_per_day=20.0,
                      food_type="rice", total_kg_food_per_day=0.3,
                      energy_source="wind", laptop_hours_per_day=3.0,
                      shower_frequency="daily",
                      flights_per_year=0, avg_km_per_flight=0.0, energy_efficient=False),
    ),
    dict(
        id="two_flights_stated",
        query=("I take the bus 10km every day. I eat nuts every day. Grid "
               "power in the US, desktop 5 hours a day, shower daily. I flew "
               "for 2 round trips this year, each one 5000km."),
        expected=dict(transport_type="bus", km_per_day=10.0,
                      food_type="nuts", meals_with_this_food_per_week=7,
                      energy_source="grid_us", desktop_hours_per_day=5.0,
                      shower_frequency="daily",
                      flights_per_year=2, avg_km_per_flight=5000.0, energy_efficient=False),
    ),
]


def get_predict_step(result):
    for step in result.get("steps", []):
        if step["tool"] == "predict_footprint":
            return step["input"]
    return None


def field_matches(field, actual, expected):
    if field in CATEGORICAL_FIELDS:
        return str(actual).strip().lower() == str(expected).strip().lower()
    if field == "energy_efficient":
        return bool(actual) == bool(expected)
    if field in NUMERIC_FIELDS:
        try:
            actual = float(actual)
        except (TypeError, ValueError):
            return False
        if abs(expected) < 1e-9:
            return abs(actual - expected) <= ABS_TOL[field]
        return abs(actual - expected) / abs(expected) <= REL_TOL
    raise ValueError(f"Unknown field: {field}")


def run():
    print(f"[INFO] Model under test: {MODEL_NAME}")
    print(f"[INFO] Running {len(CASES)} test cases against the live Groq API...\n")

    field_hits = {f: 0 for f in CATEGORICAL_FIELDS + NUMERIC_FIELDS + ["energy_efficient"]}
    field_total = {f: 0 for f in field_hits}
    case_fully_correct = 0
    tool_not_called = 0
    agent_errors = 0
    failures = []

    for i, case in enumerate(CASES):
        if i > 0:
            time.sleep(SLEEP_BETWEEN_CASES_SECONDS)
        try:
            result = run_agent(case["query"])
        except Exception as e:
            agent_errors += 1
            failures.append((case["id"], "agent raised an exception", str(e)[:200], None))
            print(f"  [{case['id']}] ERROR  {e.__class__.__name__}: {str(e)[:120]}")
            continue

        actual = get_predict_step(result)
        if actual is None:
            tool_not_called += 1
            failures.append((case["id"], "predict_footprint was never called", None, None))
            print(f"  [{case['id']}] MISMATCH  predict_footprint not called")
            continue

        case_ok = True
        for field, expected_val in case["expected"].items():
            field_total[field] += 1
            actual_val = actual.get(field, 0 if field in NUMERIC_FIELDS else None)
            ok = field_matches(field, actual_val, expected_val)
            if ok:
                field_hits[field] += 1
            else:
                case_ok = False
                failures.append((case["id"], field, actual_val, expected_val))

        if case_ok:
            case_fully_correct += 1

        threshold_label = result.get("threshold", "UNKNOWN")
        actual_co2 = result.get("actual_co2")
        print(f"  [{case['id']}] {'OK' if case_ok else 'MISMATCH'}  "
              f"actual_co2={actual_co2}  threshold={threshold_label}")

    print("\n" + "=" * 74)
    print("PER-FIELD ACCURACY")
    print("=" * 74)
    for field in field_hits:
        total = field_total[field]
        if total == 0:
            continue
        acc = field_hits[field] / total * 100
        print(f"  {field:<32} {field_hits[field]:>3}/{total:<3}  {acc:5.1f}%")

    overall = case_fully_correct / len(CASES) * 100
    print("\n" + "=" * 74)
    print("KEY NUMBERS FOR PAPER")
    print("=" * 74)
    print(f"  Test cases:                   {len(CASES)}")
    print(f"  Agent/tool-call errors:       {agent_errors}")
    print(f"  predict_footprint not called: {tool_not_called}")
    print(f"  Fully correct (all fields):   {case_fully_correct}/{len(CASES)} ({overall:.1f}%)")
    scored = [field_hits[f] / field_total[f] for f in field_hits if field_total[f]]
    if scored:
        print(f"  Mean per-field accuracy:      {mean(scored) * 100:.1f}%")

    if failures:
        print("\n" + "=" * 74)
        print("FAILURE DETAIL (for manual review)")
        print("=" * 74)
        for case_id, field, actual_val, expected_val in failures:
            print(f"  [{case_id}] {field}: actual={actual_val!r} expected={expected_val!r}")


if __name__ == "__main__":
    run()
