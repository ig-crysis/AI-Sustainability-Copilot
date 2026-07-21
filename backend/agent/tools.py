import os
import warnings
warnings.filterwarnings("ignore", message=".*Falling back to prediction.*")

import joblib
import httpx
import pandas as pd
import numpy as np
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Load trained model + preprocessors ──────────────────────────────────────
MODEL_PATH    = os.path.join("model", "artifacts", "best_model.pkl")
ENCODERS_PATH = os.path.join("data", "processed", "encoders.pkl")
SCALER_PATH   = os.path.join("data", "processed", "scaler.pkl")
IGES_PATH     = os.path.join("data", "raw", "IGES_GHG_Emissions_DB.xlsx")
OWID_PATH     = os.path.join("data", "raw", "owid_co2.csv")

model    = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
scaler   = joblib.load(SCALER_PATH)

try:
    model.set_params(device="cpu")
except Exception:
    pass

CO2SIGNAL_API_KEY = os.getenv("CO2SIGNAL_API_KEY", "")

# ── IPCC AR6 Transport emission factors (kg CO2/km) ─────────────────────────
TRANSPORT_EF = {
    "car_petrol":   0.192, "car_diesel": 0.171, "car_ev":       0.053,
    "bus":          0.089, "train":      0.041, "flight_short": 0.255,
    "flight_long":  0.195, "motorcycle": 0.114, "bicycle":      0.0,
    "walking":      0.0,
}

# ── Real food emission factors from Poore & Nemecek (kg CO2e/kg food) ───────
def _load_food_ef() -> dict:
    food_ef_path = os.path.join("data", "processed", "food_ef_real.pkl")
    if os.path.exists(food_ef_path):
        return joblib.load(food_ef_path)
    return {
        "beef": 27.0, "lamb": 39.2, "pork": 12.1, "chicken": 6.9,
        "fish":  6.1, "eggs":  4.8, "dairy": 3.2, "rice":    4.0,
        "vegetables": 2.0, "fruits": 1.1, "legumes": 0.9, "nuts": 2.5,
        "grains": 1.6,
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

# ── Training data feature ranges (for distribution check) ───────────────────
# These come from the preprocessing output — used to detect extrapolation
TRAINING_RANGES = {
    "km_per_day":       (0.0,   400.0),
    "kg_food_per_day":  (0.0,     0.6),
    "kwh_per_day":      (0.0,    20.0),
    "flights_per_year": (0,        10),
    "flight_km_total":  (0,     20000),
}

# ── Dataset average monthly CO2 (used as ML adjustment baseline) ─────────────
DATASET_MEAN_CO2 = 190.96  # from training data target mean

# ── Country mappings ─────────────────────────────────────────────────────────
COUNTRY_ZONE = {
    "india": "IN", "us": "US", "usa": "US", "united states": "US",
    "germany": "DE", "france": "FR", "uk": "GB", "australia": "AU",
    "canada": "CA", "japan": "JP", "china": "CN", "brazil": "BR",
}

ZONE_TO_ENERGY_SOURCE = {
    "IN": "grid_india", "US": "grid_us", "DE": "grid_eu",
    "FR": "grid_eu",    "GB": "grid_eu", "AU": "grid_us",
    "CA": "grid_us",    "JP": "grid_eu",
}

# ── OWID per-capita baselines ────────────────────────────────────────────────
def _load_owid_baselines() -> dict:
    baselines = {}
    try:
        df = pd.read_csv(OWID_PATH, usecols=["country", "year", "co2_per_capita"])
        df = df.dropna(subset=["co2_per_capita"])
        latest = df.sort_values("year").groupby("country").last().reset_index()
        for _, row in latest.iterrows():
            country_key = row["country"].lower().strip()
            annual_t    = float(row["co2_per_capita"])
            monthly_kg  = round(annual_t * 1000 / 12, 2)
            baselines[country_key] = {
                "per_capita_monthly_kg":    monthly_kg,
                "per_capita_annual_tonnes": round(annual_t, 3),
                "year": int(row["year"]),
            }
    except Exception as e:
        print(f"[WARN] Could not load OWID baselines: {e}")
    return baselines

OWID_BASELINES = _load_owid_baselines()


# ── Live carbon intensity fetch ──────────────────────────────────────────────
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
    """Check if inputs are within the model's training distribution."""
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
    waste_bags_per_week: number of waste bags generated per week.
    new_clothes_per_month: number of new clothing items purchased monthly.
    grocery_bill_monthly: monthly grocery spend (USD equivalent).
    energy_efficient: true if user uses energy-efficient appliances.
    """

    # ── Step 1: Compute derived lifestyle features ───────────────────────────
    WASTE_CO2_WEEKLY = {
        "small": 0.5, "medium": 1.0, "large": 2.0, "extra large": 3.5
    }
    waste_co2_monthly    = (WASTE_CO2_WEEKLY.get(waste_bag_size.lower(), 1.0)
                            * waste_bags_per_week * 4.33)
    clothing_co2_monthly = (new_clothes_per_month * 10.0) / 12
    energy_eff_val       = 1.0 if energy_efficient else 0.0

    # ── Step 2: Live grid intensity ──────────────────────────────────────────
    zone          = COUNTRY_ZONE.get(country.lower(), "IN")
    baseline_gco2 = ENERGY_INTENSITY.get(energy_source, 708)
    live          = _fetch_live_intensity(zone, baseline_gco2)
    live_gco2     = live["intensity"]

    # ── Step 3: IPCC primary calculation using REAL quantities ───────────────
    # Uses actual physical inputs — kg_food_per_day, km_per_day, kwh_per_day
    # as stated by the user, NOT the compressed training approximations.
    # This is the ground truth physical calculation.
    transport_co2 = km_per_day * 30 * TRANSPORT_EF.get(transport_type, 0.1)
    food_co2      = kg_food_per_day * 30 * FOOD_EF.get(food_type, 3.0)
    energy_co2    = kwh_per_day * 30 * (live_gco2 / 1000)
    flight_co2    = (flight_km_total / 12) * TRANSPORT_EF["flight_long"]
    waste_co2     = waste_co2_monthly
    clothing_co2  = clothing_co2_monthly

    ipcc_total = max(
        transport_co2 + food_co2 + energy_co2 +
        flight_co2 + waste_co2 + clothing_co2,
        0
    )

    # ── Step 4: XGBoost lifestyle adjustment ─────────────────────────────────
    # XGBoost captures behavioral variance beyond the primary emission categories
    # (social activity, grocery patterns, energy efficiency habits etc.)
    # We use it as a bounded multiplier relative to the dataset mean.
    # Only applied when inputs are within the model's training distribution.
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

            df = pd.DataFrame([row_data])
            for col in cat_cols:
                if col in encoders:
                    try:
                        df[col] = encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        df[col] = 0

            df[num_cols] = scaler.transform(df[num_cols])

            import xgboost as xgb
            dmatrix    = xgb.DMatrix(df[cat_cols + num_cols])
            ml_pred_kg = float(model.get_booster().predict(dmatrix)[0])

            # Adjustment factor: how does this profile compare to dataset mean?
            # Clamped to ±15% — prevents wild extrapolation
            ml_adjustment = float(np.clip(ml_pred_kg / DATASET_MEAN_CO2, 0.85, 1.15))

        except Exception as e:
            print(f"[WARN] XGBoost adjustment failed: {e}")
            ml_adjustment = 1.0

    # ── Step 5: Final prediction ─────────────────────────────────────────────
    # IPCC physical calculation × XGBoost lifestyle adjustment
    final_co2 = max(round(ipcc_total * ml_adjustment, 2), 0)

    # ── Step 6: Live grid delta for reporting ────────────────────────────────
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
            "transport_kg":  round(transport_co2, 2),
            "food_kg":       round(food_co2, 2),
            "energy_kg":     round(energy_co2, 2),
            "flights_kg":    round(flight_co2, 2),
            "waste_kg":      round(waste_co2, 2),
            "clothing_kg":   round(clothing_co2, 2),
        },
        "data_sources": {
            "primary":      "IPCC AR6 + Poore & Nemecek 2018 emission factors",
            "adjustment":   f"XGBoost lifestyle model R²=0.829 "
                            f"({'applied' if in_range else 'skipped — inputs outside training range'})",
            "grid_data":    live["source"],
        },
        "method": (
            "IPCC primary × XGBoost lifestyle adjustment × live grid correction"
            if in_range else
            "IPCC primary × live grid correction (XGBoost skipped — out of range)"
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
        mode: {
            "co2_kg":    round(total_km * ef, 2),
            "ef_per_km": ef,
        }
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
        "total_km":      total_km,
        "days":          days,
        "scenarios":     sorted_s,
        "best_option":   best,
        "worst_option":  worst,
        "max_saving_kg": round(
            scenarios[worst]["co2_kg"] - scenarios[best]["co2_kg"], 2
        ),
        "source": "IPCC AR6 Working Group III transport emission factors",
    }


# ────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Regional baseline (OWID + IGES)
# ────────────────────────────────────────────────────────────────────────────
@tool
def get_regional_baseline(country: str) -> dict:
    """
    Get per-capita CO2 baseline for a country.
    Uses Our World in Data (latest year) as primary source,
    IGES UNFCCC database as secondary for Annex I countries.
    country: e.g. 'India', 'Germany', 'United States'.
    """
    country_lower = country.strip().lower()

    # ── Primary: OWID dataset ─────────────────────────────────────────────────
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

    # ── Secondary: IGES UNFCCC ────────────────────────────────────────────────
    POPULATION_M = {
        "germany": 83.2, "france": 67.8, "austria": 9.1, "belgium": 11.6,
        "netherlands": 17.9, "sweden": 10.5, "norway": 5.4, "finland": 5.5,
        "denmark": 5.9, "poland": 37.8, "italy": 59.2, "spain": 47.4,
        "portugal": 10.3, "czech republic": 10.9, "hungary": 9.7,
        "romania": 19.0, "bulgaria": 6.5, "croatia": 3.9,
        "united kingdom": 67.3, "australia": 25.9, "canada": 38.2,
        "japan": 125.7, "united states": 331.4,
    }

    try:
        df = pd.read_excel(IGES_PATH, sheet_name="GHG & CO2 EMISSIONS", header=0)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        match = df[df["Party"].str.strip().str.lower().str.contains(
            country_lower, na=False
        )]
        if match.empty:
            return {
                "error": f"'{country}' not found in OWID or IGES databases.",
                "hint":  "Try the full country name e.g. 'United States', 'United Kingdom'",
            }

        row      = match[match["Emission Type"] == "GHG"].iloc[0]
        yr_cols  = [c for c in df.columns
                    if isinstance(c, (int, float)) and 1990 <= c <= 2022]
        recent   = sorted(yr_cols)[-5:]
        trend    = {str(int(y)): round(float(row[y]) / 1000, 2)
                    for y in recent if pd.notna(row[y])}

        latest_yr = str(int(max(recent)))
        latest_mt = trend.get(latest_yr, 0)
        pop       = POPULATION_M.get(country_lower, 10)
        per_cap_t = round((latest_mt * 1e6) / (pop * 1e6), 2)
        monthly   = round(per_cap_t * 1000 / 12, 2)

        return {
            "country":                  country,
            "per_capita_monthly_kg":    monthly,
            "per_capita_annual_tonnes": per_cap_t,
            "data_year":                latest_yr,
            "trend_last_5_years_mt":    trend,
            "source":                   "IGES GHG Emissions DB (UNFCCC official submissions)",
        }

    except Exception as e:
        return {"error": str(e)}