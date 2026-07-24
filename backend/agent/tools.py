import os
import warnings
warnings.filterwarnings("ignore", message=".*Falling back to prediction.*")
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import httpx
import pandas as pd
import numpy as np
import xgboost as xgb
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_UBJ_PATH = os.path.join("model", "artifacts", "best_model.ubj")
MODEL_PKL_PATH = os.path.join("model", "artifacts", "best_model.pkl")
ENCODERS_PATH  = os.path.join("data", "processed", "encoders.pkl")
SCALER_PATH    = os.path.join("data", "processed", "scaler.pkl")
OWID_PKL_PATH  = os.path.join("data", "processed", "owid_baselines.pkl")
IGES_PATH      = os.path.join("data", "raw", "IGES_GHG_Emissions_DB.xlsx")

# ── Load XGBoost booster (version-independent native format) ─────────────────
def _load_booster():
    if os.path.exists(MODEL_UBJ_PATH):
        b = xgb.Booster()
        b.load_model(MODEL_UBJ_PATH)
        print(f"[INFO] Loaded XGBoost from native format: {MODEL_UBJ_PATH}")
        return b
    # Fallback to pkl with suppressed warnings
    print(f"[INFO] Loading XGBoost from pkl: {MODEL_PKL_PATH}")
    m = joblib.load(MODEL_PKL_PATH)
    try:
        m.set_params(device="cpu")
    except Exception:
        pass
    return m.get_booster()

booster  = _load_booster()
encoders = joblib.load(ENCODERS_PATH)
scaler   = joblib.load(SCALER_PATH)

CO2SIGNAL_API_KEY = os.getenv("CO2SIGNAL_API_KEY", "")

# ── IPCC AR6 Transport emission factors (kg CO2/km) ─────────────────────────
TRANSPORT_EF = {
    "car_petrol":   0.192, "car_diesel": 0.171, "car_ev":       0.053,
    "bus":          0.089, "train":      0.041, "flight_short": 0.255,
    "flight_long":  0.195, "motorcycle": 0.114, "bicycle":      0.0,
    "walking":      0.0,
}

# ── Food emission factors from Poore & Nemecek 2018 ─────────────────────────
def _load_food_ef() -> dict:
    food_ef_path = os.path.join("data", "processed", "food_ef_real.pkl")
    if os.path.exists(food_ef_path):
        return joblib.load(food_ef_path)
    return {
        "beef": 27.0, "lamb": 39.2, "pork": 12.1, "chicken": 6.9,
        "fish":  6.1, "eggs":  4.8, "dairy": 3.2, "rice":    4.0,
        "vegetables": 2.0, "fruits": 1.1, "legumes": 0.9,
        "nuts": 2.5, "grains": 1.6,
    }

FOOD_EF = _load_food_ef()

# ── Grid carbon intensity baselines (gCO2/kWh) ──────────────────────────────
ENERGY_INTENSITY = {
    "coal":        820, "natural_gas": 490, "oil":     650,
    "solar":        41, "wind":         11, "hydro":    24,
    "nuclear":      12, "grid_india":  708, "grid_us": 386,
    "grid_eu":     276,
}
ENERGY_EF = {k: v / 1000 for k, v in ENERGY_INTENSITY.items()}

# ── Training distribution ranges ─────────────────────────────────────────────
TRAINING_RANGES = {
    "km_per_day":       (0.0,  400.0),
    "kg_food_per_day":  (0.0,    0.6),
    "kwh_per_day":      (0.0,   20.0),
    "flights_per_year": (0,       10),
    "flight_km_total":  (0,    20000),
}
DATASET_MEAN_CO2 = 190.96

# ── Country mappings ─────────────────────────────────────────────────────────
COUNTRY_ZONE = {
    "india": "IN", "us": "US", "usa": "US", "united states": "US",
    "germany": "DE", "france": "FR", "uk": "GB", "australia": "AU",
    "canada": "CA", "japan": "JP", "china": "CN", "brazil": "BR",
}

# ── OWID baselines — pkl first, then hardcoded fallback ──────────────────────
def _load_owid_baselines() -> dict:
    if os.path.exists(OWID_PKL_PATH):
        try:
            data = joblib.load(OWID_PKL_PATH)
            print(f"[INFO] Loaded OWID baselines: {len(data)} countries")
            return data
        except Exception as e:
            print(f"[WARN] Could not load owid_baselines.pkl: {e}")

    # Hardcoded fallback for common countries (IEA/GCP 2023)
    print("[WARN] Using hardcoded OWID fallback")
    return {
        "india":          {"per_capita_monthly_kg": 183.42,  "per_capita_annual_tonnes": 2.201, "year": 2023},
        "china":          {"per_capita_monthly_kg": 583.33,  "per_capita_annual_tonnes": 7.0,   "year": 2023},
        "united states":  {"per_capita_monthly_kg": 1191.67, "per_capita_annual_tonnes": 14.3,  "year": 2023},
        "germany":        {"per_capita_monthly_kg": 564.08,  "per_capita_annual_tonnes": 6.769, "year": 2023},
        "united kingdom": {"per_capita_monthly_kg": 416.67,  "per_capita_annual_tonnes": 5.0,   "year": 2023},
        "france":         {"per_capita_monthly_kg": 375.0,   "per_capita_annual_tonnes": 4.5,   "year": 2023},
        "australia":      {"per_capita_monthly_kg": 1166.67, "per_capita_annual_tonnes": 14.0,  "year": 2023},
        "canada":         {"per_capita_monthly_kg": 1291.67, "per_capita_annual_tonnes": 15.5,  "year": 2023},
        "japan":          {"per_capita_monthly_kg": 691.67,  "per_capita_annual_tonnes": 8.3,   "year": 2023},
        "brazil":         {"per_capita_monthly_kg": 183.33,  "per_capita_annual_tonnes": 2.2,   "year": 2023},
        "south africa":   {"per_capita_monthly_kg": 508.33,  "per_capita_annual_tonnes": 6.1,   "year": 2023},
        "indonesia":      {"per_capita_monthly_kg": 175.0,   "per_capita_annual_tonnes": 2.1,   "year": 2023},
        "pakistan":       {"per_capita_monthly_kg": 91.67,   "per_capita_annual_tonnes": 1.1,   "year": 2023},
        "bangladesh":     {"per_capita_monthly_kg": 41.67,   "per_capita_annual_tonnes": 0.5,   "year": 2023},
        "russia":         {"per_capita_monthly_kg": 1000.0,  "per_capita_annual_tonnes": 12.0,  "year": 2023},
        "italy":          {"per_capita_monthly_kg": 475.0,   "per_capita_annual_tonnes": 5.7,   "year": 2023},
        "spain":          {"per_capita_monthly_kg": 433.33,  "per_capita_annual_tonnes": 5.2,   "year": 2023},
        "sweden":         {"per_capita_monthly_kg": 333.33,  "per_capita_annual_tonnes": 4.0,   "year": 2023},
        "norway":         {"per_capita_monthly_kg": 750.0,   "per_capita_annual_tonnes": 9.0,   "year": 2023},
        "saudi arabia":   {"per_capita_monthly_kg": 1283.33, "per_capita_annual_tonnes": 15.4,  "year": 2023},
        "uae":            {"per_capita_monthly_kg": 1750.0,  "per_capita_annual_tonnes": 21.0,  "year": 2023},
    }

OWID_BASELINES = _load_owid_baselines()


# ── Live carbon intensity ─────────────────────────────────────────────────────
def _fetch_live_intensity(zone: str, baseline: float) -> dict:
    if CO2SIGNAL_API_KEY:
        for url, headers in [
            (f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}",
             {"auth-token": CO2SIGNAL_API_KEY}),
            (f"https://api.co2signal.com/v1/latest?countryCode={zone}",
             {"auth-token": CO2SIGNAL_API_KEY}),
        ]:
            try:
                resp = httpx.get(url, headers=headers, timeout=6)
                if resp.status_code == 200 and resp.text.strip():
                    data      = resp.json()
                    intensity = (data.get("carbonIntensity") or
                                 data.get("data", {}).get("carbonIntensity"))
                    if intensity:
                        return {"intensity": float(intensity), "source": "live"}
            except Exception:
                continue
    return {"intensity": baseline, "source": "static_fallback"}


def _check_in_training_range(km_per_day, kg_food_per_day,
                              kwh_per_day, flights_per_year,
                              flight_km_total) -> bool:
    checks = [
        TRAINING_RANGES["km_per_day"][0] <= km_per_day <= TRAINING_RANGES["km_per_day"][1],
        TRAINING_RANGES["kg_food_per_day"][0] <= kg_food_per_day <= TRAINING_RANGES["kg_food_per_day"][1],
        TRAINING_RANGES["kwh_per_day"][0] <= kwh_per_day <= TRAINING_RANGES["kwh_per_day"][1],
        TRAINING_RANGES["flights_per_year"][0] <= flights_per_year <= TRAINING_RANGES["flights_per_year"][1],
        TRAINING_RANGES["flight_km_total"][0] <= flight_km_total <= TRAINING_RANGES["flight_km_total"][1],
    ]
    return all(checks)


# ────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Predict carbon footprint
# ────────────────────────────────────────────────────────────────────────────
@tool
def predict_footprint(
    transport_type: str,
    km_per_day: float,
    food_type: str,
    kg_food_per_day: float,
    energy_source: str,
    kwh_per_day: float,
    flights_per_year: int,
    flight_km_total: float,
    country: str = "india",
    waste_bag_size: str = "medium",
    waste_bags_per_week: float = 1.0,
    new_clothes_per_month: float = 2.0,
    grocery_bill_monthly: float = 200.0,
    energy_efficient: bool = False,
) -> dict:
    """
    Predict monthly carbon footprint (kg CO2) using a hybrid pipeline:
    IPCC emission factors as physical ground truth (primary),
    XGBoost lifestyle adjustment factor (secondary),
    live grid carbon intensity for real-time energy correction.
    transport_type: car_petrol, car_diesel, car_ev, bus, train,
                    flight_short, flight_long, motorcycle, bicycle, walking.
    food_type: beef, lamb, pork, chicken, fish, eggs, dairy,
               rice, vegetables, fruits, legumes, nuts, grains.
    energy_source: coal, natural_gas, oil, solar, wind, hydro,
                   nuclear, grid_india, grid_us, grid_eu.
    country: user country for live grid e.g. 'india', 'germany'.
    waste_bag_size: small, medium, large, extra large.
    waste_bags_per_week: number of waste bags per week.
    new_clothes_per_month: number of new clothing items monthly.
    grocery_bill_monthly: monthly grocery spend USD equivalent.
    energy_efficient: true if user uses energy-efficient appliances.
    """
    WASTE_CO2_WEEKLY = {
        "small": 0.5, "medium": 1.0, "large": 2.0, "extra large": 3.5
    }
    waste_co2_monthly    = (WASTE_CO2_WEEKLY.get(waste_bag_size.lower(), 1.0)
                            * waste_bags_per_week * 4.33)
    clothing_co2_monthly = (new_clothes_per_month * 10.0) / 12
    energy_eff_val       = 1.0 if energy_efficient else 0.0

    zone          = COUNTRY_ZONE.get(country.lower(), "IN")
    baseline_gco2 = ENERGY_INTENSITY.get(energy_source, 708)
    live          = _fetch_live_intensity(zone, baseline_gco2)
    live_gco2     = live["intensity"]

    # IPCC primary calculation
    transport_co2 = km_per_day * 30 * TRANSPORT_EF.get(transport_type, 0.1)
    food_co2      = kg_food_per_day * 30 * FOOD_EF.get(food_type, 3.0)
    energy_co2    = kwh_per_day * 30 * (live_gco2 / 1000)
    flight_co2    = (flight_km_total / 12) * TRANSPORT_EF["flight_long"]
    waste_co2     = waste_co2_monthly
    clothing_co2  = clothing_co2_monthly
    ipcc_total    = max(transport_co2 + food_co2 + energy_co2 +
                        flight_co2 + waste_co2 + clothing_co2, 0)

    # XGBoost lifestyle adjustment
    ml_adjustment = 1.0
    ml_pred_kg    = None
    in_range      = _check_in_training_range(
        km_per_day, kg_food_per_day, kwh_per_day,
        flights_per_year, flight_km_total
    )

    if in_range:
        try:
            cat_cols = ["transport_type", "food_type", "energy_source"]
            num_cols = ["km_per_day", "kg_food_per_day", "kwh_per_day",
                        "flights_per_year", "flight_km_total",
                        "waste_co2_monthly", "clothing_co2_monthly",
                        "grocery_bill_monthly", "energy_efficient"]

            row_data = {
                "transport_type":        transport_type,
                "food_type":             food_type,
                "energy_source":         energy_source,
                "km_per_day":            km_per_day,
                "kg_food_per_day":       kg_food_per_day,
                "kwh_per_day":           kwh_per_day,
                "flights_per_year":      flights_per_year,
                "flight_km_total":       float(flight_km_total),
                "waste_co2_monthly":     waste_co2_monthly,
                "clothing_co2_monthly":  clothing_co2_monthly,
                "grocery_bill_monthly":  grocery_bill_monthly,
                "energy_efficient":      energy_eff_val,
            }

            df_pred = pd.DataFrame([row_data])
            for col in cat_cols:
                if col in encoders:
                    try:
                        df_pred[col] = encoders[col].transform(
                            df_pred[col].astype(str))
                    except ValueError:
                        df_pred[col] = 0

            df_pred[num_cols] = scaler.transform(df_pred[num_cols])
            dmatrix    = xgb.DMatrix(df_pred[cat_cols + num_cols])
            ml_pred_kg = float(booster.predict(dmatrix)[0])
            ml_adjustment = float(np.clip(
                ml_pred_kg / DATASET_MEAN_CO2, 0.85, 1.15
            ))
        except Exception as e:
            print(f"[WARN] XGBoost adjustment failed: {e}")
            ml_adjustment = 1.0

    final_co2    = max(round(ipcc_total * ml_adjustment, 2), 0)
    energy_delta = round(
        kwh_per_day * 30 * ((live_gco2 - baseline_gco2) / 1000), 2
    )

    return {
        "predicted_monthly_co2_kg": final_co2,
        "ipcc_baseline_kg":         round(ipcc_total, 2),
        "ml_adjustment_factor":     round(ml_adjustment, 4),
        "ml_in_training_range":     in_range,
        "live_grid_delta_kg":       energy_delta,
        "live_intensity_gco2_kwh":  round(live_gco2, 1),
        "annual_estimate_kg":       round(final_co2 * 12, 2),
        "breakdown": {
            "transport_kg": round(transport_co2, 2),
            "food_kg":      round(food_co2, 2),
            "energy_kg":    round(energy_co2, 2),
            "flights_kg":   round(flight_co2, 2),
            "waste_kg":     round(waste_co2, 2),
            "clothing_kg":  round(clothing_co2, 2),
        },
        "data_sources": {
            "primary":   "IPCC AR6 + Poore & Nemecek 2018 emission factors",
            "adjustment": f"XGBoost R²=0.829 "
                          f"({'applied' if in_range else 'skipped-out of range'})",
            "grid_data": live["source"],
        },
        "method": (
            "IPCC primary × XGBoost lifestyle adjustment × live grid correction"
            if in_range else
            "IPCC primary × live grid correction (XGBoost skipped)"
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Live carbon intensity
# ────────────────────────────────────────────────────────────────────────────
@tool
def get_live_carbon_intensity(country: str) -> dict:
    """
    Fetch real-time electricity carbon intensity (gCO2/kWh) for a country.
    Falls back to verified static values if live API unavailable.
    country: e.g. 'india', 'germany', 'us'.
    """
    STATIC_INTENSITY = {
        "IN": 708, "US": 386, "DE": 276, "FR": 85,
        "GB": 233, "AU": 480, "CN": 555, "JP": 462,
        "CA": 130, "BR": 100,
    }
    zone     = COUNTRY_ZONE.get(country.lower(), "IN")
    baseline = STATIC_INTENSITY.get(zone, 400)
    live     = _fetch_live_intensity(zone, baseline)

    return {
        "country":                   country,
        "zone":                      zone,
        "carbon_intensity_gco2_kwh": round(live["intensity"], 1),
        "per_kwh_kg":                round(live["intensity"] / 1000, 4),
        "source":                    live["source"],
        "note": ("Live Electricity Maps data" if live["source"] == "live"
                 else "IPCC/IEA verified static value"),
    }


# ────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Compare transport scenarios
# ────────────────────────────────────────────────────────────────────────────
@tool
def compare_transport_scenarios(km_per_day: float, days: int = 30) -> dict:
    """
    Compare CO2 emissions across all transport modes for a given daily distance.
    km_per_day: daily travel distance in km.
    days: number of days (default 30).
    """
    total_km  = km_per_day * days
    scenarios = {
        mode: {"co2_kg": round(total_km * ef, 2), "ef_per_km": ef}
        for mode, ef in TRANSPORT_EF.items()
    }
    sorted_s = dict(sorted(scenarios.items(), key=lambda x: x[1]["co2_kg"]))
    best     = list(sorted_s.keys())[0]
    worst    = list(sorted_s.keys())[-1]
    for mode in sorted_s:
        sorted_s[mode]["saving_vs_worst_kg"] = round(
            scenarios[worst]["co2_kg"] - scenarios[mode]["co2_kg"], 2
        )
    return {
        "total_km": total_km, "days": days, "scenarios": sorted_s,
        "best_option": best, "worst_option": worst,
        "max_saving_kg": round(
            scenarios[worst]["co2_kg"] - scenarios[best]["co2_kg"], 2),
        "source": "IPCC AR6 Working Group III transport emission factors",
    }


# ────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Regional baseline (OWID + hardcoded fallback)
# ────────────────────────────────────────────────────────────────────────────
@tool
def get_regional_baseline(country: str) -> dict:
    """
    Get per-capita CO2 baseline for a country.
    Uses Our World in Data (latest year) as primary source.
    country: e.g. 'India', 'Germany', 'United States'.
    """
    country_lower = country.strip().lower()

    if country_lower in OWID_BASELINES:
        b = OWID_BASELINES[country_lower]
        return {
            "country":                  country,
            "per_capita_monthly_kg":    b["per_capita_monthly_kg"],
            "per_capita_annual_tonnes": b["per_capita_annual_tonnes"],
            "data_year":                b["year"],
            "source":                   "Our World in Data (OWID) — based on GCP",
            "note": f"Latest available year: {b['year']}",
        }

    return {
        "error": f"'{country}' not found in baselines database.",
        "hint":  "Try: India, Germany, United States, China, Brazil, Japan",
    }