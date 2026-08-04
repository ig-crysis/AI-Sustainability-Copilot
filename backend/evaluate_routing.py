"""
Routing/agreement study: how often does XGBoost actually fire on realistic
queries, and when it does, how much does it disagree with the IPCC formula?

evaluate_ablation.py already validates XGBoost against the synthetic held-out
test set (rows sampled from the training distribution, so they're
near-guaranteed to pass the in-training-range gate). It does NOT tell you
what happens on the kind of input an actual chat user types — LIMITATIONS.md
already asserts, qualitatively, that "most direct household-kWh statements
will legitimately route to the IPCC fallback" and "many otherwise-normal
queries... will fall back to the physical formula", but nothing has measured
HOW OFTEN, or how much the two methods disagree when XGBoost does fire. This
script measures both, over randomly-sampled realistic scenarios (not dataset
rows).

Faithful to production: imports the actual loaded encoders/scaler/booster and
emission-factor constants from agent/tools.py (not copies), so this is the
real model, not a re-implementation. The one deliberate difference from
agent/tools.py::predict_footprint is that live grid intensity is held equal
to each energy source's static baseline (no httpx call) -- same
simplification evaluate_ablation.py's main table already uses, for the same
reason: there's no "live" value to fetch for a hand-built scenario, and this
keeps the script network-free and reproducible run-to-run.

Run: python evaluate_routing.py
"""

import sys
import random

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, ".")
from agent.tools import (  # noqa: E402
    booster, encoders, scaler,
    TRANSPORT_EF, FOOD_EF, ENERGY_INTENSITY, ENERGY_EF,
    TRAINING_RANGES, DEVICE_KWH_PER_HR, SHOWER_KWH,
    BASE_HOUSEHOLD_KWH, MEAL_KG,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 42
N_SCENARIOS = 3000

TRANSPORT_TYPES = list(TRANSPORT_EF.keys())
FOOD_TYPES = list(FOOD_EF.keys())
ENERGY_SOURCES = list(ENERGY_INTENSITY.keys())
SHOWER_FREQS = list(SHOWER_KWH.keys())

# Fixed at production defaults -- not the subject of this study, and this is
# what predict_footprint itself defaults to when a user doesn't mention them.
WASTE_CO2_MONTHLY = 1.0 * 1.0 * 4.33      # medium bag, 1/week
CLOTHING_CO2_MONTHLY = (2.0 * 10.0) / 12  # 2 new items/month
GROCERY_BILL_MONTHLY = 200.0
ENERGY_EFFICIENT_VAL = 0.0

CAT_COLS = ["transport_type", "food_type", "energy_source"]
NUM_COLS = [
    "km_per_day", "kg_food_per_day", "kwh_per_day",
    "flights_per_year", "flight_km_total",
    "waste_co2_monthly", "clothing_co2_monthly",
    "grocery_bill_monthly", "energy_efficient",
]


def sample_scenario(rng: random.Random) -> dict:
    """One randomly-sampled, realistic-style user scenario -- atomic facts
    only, mirroring what the extraction LLM would actually pass to
    predict_footprint (see EXTRACTION_SYSTEM_PROMPT in agent/carbon_agent.py).
    """
    transport_type = rng.choice(TRANSPORT_TYPES)
    km_per_day = rng.uniform(0, 400) if rng.random() < 0.9 else rng.uniform(0, 40)

    food_type = rng.choice(FOOD_TYPES)
    if rng.random() < 0.5:
        meals_with_this_food_per_week = rng.uniform(0, 7)
        total_kg_food_per_day = 0.0
    else:
        meals_with_this_food_per_week = 0.0
        total_kg_food_per_day = rng.uniform(0.05, 1.5)

    energy_source = rng.choice(ENERGY_SOURCES)
    shower_frequency = rng.choice(SHOWER_FREQS)
    if rng.random() < 0.5:
        phone_hours_per_day = rng.uniform(0, 8)
        laptop_hours_per_day = rng.uniform(0, 10)
        desktop_hours_per_day = rng.uniform(0, 10)
        tv_hours_per_day = rng.uniform(0, 6)
        total_kwh_per_day = 0.0
    else:
        phone_hours_per_day = laptop_hours_per_day = 0.0
        desktop_hours_per_day = tv_hours_per_day = 0.0
        total_kwh_per_day = rng.uniform(1, 30)

    if rng.random() < 0.6:
        flights_per_year, avg_km_per_flight = 0, 0.0
    else:
        flights_per_year = rng.randint(1, 6)
        avg_km_per_flight = rng.uniform(300, 15000)

    return dict(
        transport_type=transport_type, km_per_day=km_per_day,
        food_type=food_type,
        meals_with_this_food_per_week=meals_with_this_food_per_week,
        total_kg_food_per_day=total_kg_food_per_day,
        energy_source=energy_source, shower_frequency=shower_frequency,
        phone_hours_per_day=phone_hours_per_day,
        laptop_hours_per_day=laptop_hours_per_day,
        desktop_hours_per_day=desktop_hours_per_day,
        tv_hours_per_day=tv_hours_per_day,
        total_kwh_per_day=total_kwh_per_day,
        flights_per_year=flights_per_year, avg_km_per_flight=avg_km_per_flight,
    )


def derive_features(s: dict) -> dict:
    """Atomic facts -> the 5 numeric features the range gate and the model
    actually see. Formula copied verbatim from agent/tools.py::predict_footprint."""
    if s["total_kg_food_per_day"] > 0:
        kg_food_per_day = round(s["total_kg_food_per_day"], 4)
    else:
        kg_food_per_day = round(
            max(min(s["meals_with_this_food_per_week"], 7.0), 0.0) / 7.0 * MEAL_KG, 4
        )

    if s["total_kwh_per_day"] > 0:
        kwh_per_day = round(s["total_kwh_per_day"], 3)
    else:
        kwh_per_day = round(
            s["phone_hours_per_day"] * DEVICE_KWH_PER_HR["phone"]
            + s["laptop_hours_per_day"] * DEVICE_KWH_PER_HR["laptop"]
            + s["desktop_hours_per_day"] * DEVICE_KWH_PER_HR["desktop"]
            + s["tv_hours_per_day"] * DEVICE_KWH_PER_HR["tv"]
            + SHOWER_KWH.get(s["shower_frequency"], 0.9)
            + BASE_HOUSEHOLD_KWH,
            3,
        )

    flight_km_total = float(s["flights_per_year"]) * float(s["avg_km_per_flight"])

    return dict(
        km_per_day=s["km_per_day"], kg_food_per_day=kg_food_per_day,
        kwh_per_day=kwh_per_day, flights_per_year=s["flights_per_year"],
        flight_km_total=flight_km_total,
    )


def in_range_per_feature(feats: dict) -> dict:
    """Per-feature pass/fail against TRAINING_RANGES (not just the overall
    AND-gate) so we can see which single feature most often causes a
    fallback -- this is the empirical version of the "single AND-gate"
    limitation already noted qualitatively in research/LIMITATIONS.md."""
    return {
        name: (TRAINING_RANGES[name][0] <= feats[name] <= TRAINING_RANGES[name][1])
        for name in TRAINING_RANGES
    }


def categorical_in_vocab(s: dict) -> dict:
    """Per-category vocabulary coverage check, mirroring the fix in
    agent/tools.py::_check_categorical_in_vocab. The raw training data only
    has a handful of category labels per column (see
    data_preprocessing.py's DIET_TO_FOOD_EXACT / HEATING_MAP), far fewer than
    predict_footprint's Literal type hints expose -- e.g. "grid_india"
    (predict_footprint's own default energy_source) is not in the trained
    vocabulary at all."""
    return {
        "transport_type": s["transport_type"] in encoders["transport_type"].classes_,
        "food_type": s["food_type"] in encoders["food_type"].classes_,
        "energy_source": s["energy_source"] in encoders["energy_source"].classes_,
    }


def ipcc_breakdown_total(s: dict, feats: dict) -> float:
    """Static-grid IPCC total -- same formula as agent/tools.py, with live
    grid intensity == the energy source's own baseline (no httpx call; see
    module docstring)."""
    baseline_gco2 = ENERGY_INTENSITY.get(s["energy_source"], 708)
    transport_co2 = feats["km_per_day"] * 30 * TRANSPORT_EF.get(s["transport_type"], 0.1)
    food_co2 = feats["kg_food_per_day"] * 30 * FOOD_EF.get(s["food_type"], 3.0)
    energy_co2 = feats["kwh_per_day"] * 30 * (baseline_gco2 / 1000)
    flight_co2 = (feats["flight_km_total"] / 12) * TRANSPORT_EF["flight_long"]
    return max(
        transport_co2 + food_co2 + energy_co2 + flight_co2
        + WASTE_CO2_MONTHLY + CLOTHING_CO2_MONTHLY,
        0,
    )


def xgboost_prediction(s: dict, feats: dict) -> float:
    """Same preprocessing + prediction pipeline as agent/tools.py::predict_footprint,
    using the actual loaded encoders/scaler/booster (not copies)."""
    row_data = {
        "transport_type": s["transport_type"], "food_type": s["food_type"],
        "energy_source": s["energy_source"],
        "km_per_day": feats["km_per_day"], "kg_food_per_day": feats["kg_food_per_day"],
        "kwh_per_day": feats["kwh_per_day"],
        "flights_per_year": feats["flights_per_year"],
        "flight_km_total": float(feats["flight_km_total"]),
        "waste_co2_monthly": WASTE_CO2_MONTHLY,
        "clothing_co2_monthly": CLOTHING_CO2_MONTHLY,
        "grocery_bill_monthly": GROCERY_BILL_MONTHLY,
        "energy_efficient": ENERGY_EFFICIENT_VAL,
    }
    df_pred = pd.DataFrame([row_data])
    for col in CAT_COLS:
        try:
            df_pred[col] = encoders[col].transform(df_pred[col].astype(str))
        except ValueError:
            df_pred[col] = 0
    df_pred[NUM_COLS] = scaler.transform(df_pred[NUM_COLS])
    dmatrix = xgb.DMatrix(df_pred[CAT_COLS + NUM_COLS])
    return float(booster.predict(dmatrix)[0])


def run():
    rng = random.Random(SEED)
    print(f"[INFO] Sampling {N_SCENARIOS} realistic scenarios (seed={SEED})...")

    in_range_count = 0            # numeric ranges only (original measure)
    true_in_range_count = 0       # numeric ranges AND categorical vocabulary (post-fix gate)
    feature_fail_counts = {name: 0 for name in TRAINING_RANGES}
    cat_fail_counts = {name: 0 for name in ("transport_type", "food_type", "energy_source")}
    agreements = []  # (ml_pred, ipcc_pred) for true-in-range scenarios
    all_ipcc = []

    for _ in range(N_SCENARIOS):
        s = sample_scenario(rng)
        feats = derive_features(s)
        per_feature = in_range_per_feature(feats)
        numeric_in_range = all(per_feature.values())
        per_cat = categorical_in_vocab(s)
        cat_in_vocab = all(per_cat.values())
        true_in_range = numeric_in_range and cat_in_vocab

        ipcc_pred = ipcc_breakdown_total(s, feats)
        all_ipcc.append(ipcc_pred)

        if numeric_in_range:
            in_range_count += 1
        else:
            for name, ok in per_feature.items():
                if not ok:
                    feature_fail_counts[name] += 1

        if not cat_in_vocab:
            for name, ok in per_cat.items():
                if not ok:
                    cat_fail_counts[name] += 1

        if true_in_range:
            true_in_range_count += 1
            ml_pred = xgboost_prediction(s, feats)
            agreements.append((ml_pred, ipcc_pred))

    n = N_SCENARIOS
    pct_in_range = in_range_count / n * 100
    pct_true_in_range = true_in_range_count / n * 100

    print("\n" + "=" * 74)
    print("ROUTING: HOW OFTEN DOES XGBOOST ACTUALLY FIRE ON REALISTIC QUERIES?")
    print("=" * 74)
    print(f"  Scenarios sampled:                          {n}")
    print(f"  Numeric ranges only (old, pre-fix measure): {in_range_count}/{n} ({pct_in_range:.1f}%)")
    print(f"  Numeric ranges + categorical vocabulary:    {true_in_range_count}/{n} ({pct_true_in_range:.1f}%)")
    print(f"  Out-of-range (IPCC fallback used):          {n - true_in_range_count}/{n} ({100 - pct_true_in_range:.1f}%)")
    print("  (Before the categorical-vocabulary fix, the gap between the two rows above")
    print("   was invisible: unseen categories silently fell through to `except: =0`")
    print("   inside predict_footprint instead of triggering the IPCC fallback.)")

    print("\n" + "-" * 74)
    print("  Numeric-range fallbacks: which feature caused it?")
    print("  (a scenario can fail more than one feature, so these don't need to sum to the total)")
    print("-" * 74)
    for name, count in sorted(feature_fail_counts.items(), key=lambda x: -x[1]):
        pct_of_out_of_range = (count / (n - in_range_count) * 100) if in_range_count < n else 0.0
        print(f"  {name:<20} {count:>4} scenarios ({pct_of_out_of_range:5.1f}% of all numeric-out-of-range cases)")

    print("\n" + "-" * 74)
    print("  Categorical-vocabulary fallbacks: which field caused it?")
    print("  (out of the vocabulary the raw training data actually contains -- see")
    print("   data_preprocessing.py's DIET_TO_FOOD_EXACT / HEATING_MAP / VEHICLE_MAP)")
    print("-" * 74)
    for name, count in sorted(cat_fail_counts.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        print(f"  {name:<20} {count:>4} scenarios ({pct:5.1f}% of all {n} scenarios)")
    for name, le in encoders.items():
        print(f"    trained vocabulary for {name}: {sorted(le.classes_)}")

    if agreements:
        ml_arr = np.array([a[0] for a in agreements])
        ipcc_arr = np.array([a[1] for a in agreements])
        abs_diff = np.abs(ml_arr - ipcc_arr)
        # relative diff vs the larger of the two, to avoid blowing up near zero
        denom = np.maximum(np.maximum(ml_arr, ipcc_arr), 1.0)
        rel_diff = abs_diff / denom * 100

        print("\n" + "=" * 74)
        print("AGREEMENT: WHEN XGBOOST FIRES, HOW MUCH DOES IT DIFFER FROM IPCC?")
        print("=" * 74)
        print("  (Not a ground-truth comparison -- both methods estimate the same")
        print("   quantity differently. Large disagreement means the routing gate's")
        print("   in/out-of-range decision has a big practical effect on the number")
        print("   the user sees, which is the point of measuring it.)")
        print(f"  In-range scenarios:               {len(agreements)}")
        print(f"  Mean XGBoost prediction:          {ml_arr.mean():.2f} kg CO2/month")
        print(f"  Mean IPCC prediction:             {ipcc_arr.mean():.2f} kg CO2/month")
        print(f"  Mean absolute difference:         {abs_diff.mean():.2f} kg CO2/month")
        print(f"  Median absolute difference:       {np.median(abs_diff):.2f} kg CO2/month")
        print(f"  Mean relative difference:         {rel_diff.mean():.1f}%")
        print(f"  Scenarios differing by >50%:      {(rel_diff > 50).sum()}/{len(agreements)} "
              f"({(rel_diff > 50).sum() / len(agreements) * 100:.1f}%)")

    print("\n" + "=" * 74)
    print("KEY NUMBERS FOR PAPER")
    print("=" * 74)
    print(f"  % of realistic queries where XGBoost fires, numeric ranges only (pre-fix): {pct_in_range:.1f}%")
    print(f"  % of realistic queries where XGBoost fires, numeric + categorical (post-fix): {pct_true_in_range:.1f}%")
    if agreements:
        print(f"  Mean disagreement (XGBoost vs IPCC) when both apply:  "
              f"{abs_diff.mean():.2f} kg CO2/month ({rel_diff.mean():.1f}% relative)")
    top_blocker = max(feature_fail_counts, key=feature_fail_counts.get)
    print(f"  Single biggest numeric-range cause of fallback:         {top_blocker} "
          f"({feature_fail_counts[top_blocker]} scenarios)")
    top_cat_blocker = max(cat_fail_counts, key=cat_fail_counts.get)
    print(f"  Single biggest categorical cause of fallback:           {top_cat_blocker} "
          f"({cat_fail_counts[top_cat_blocker]} scenarios, {cat_fail_counts[top_cat_blocker]/n*100:.1f}%)")


if __name__ == "__main__":
    run()
