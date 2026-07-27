"""
Hybrid Fusion Layer
===================
Combines XGBoost ML prediction (trained on a synthetically-generated
dataset — see research/LIMITATIONS.md) with live carbon intensity from
Electricity Maps API to produce a calibrated CO2 estimate.

IEEE Novelty: Static ML baseline × dynamic grid adjustment
= more accurate than either alone.
"""

import os
import httpx
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Baseline grid intensities the model was trained on (gCO2/kWh) ───────────
BASELINE_INTENSITY = {
    "grid_india": 708, "grid_us":  386, "grid_eu":  276,
    "coal":       820, "natural_gas": 490,
    "solar":       41, "wind":      11, "hydro":     24,
    "nuclear":     12,
}

COUNTRY_ZONE = {
    "india": "IN", "us": "US", "usa": "US", "germany": "DE",
    "france": "FR", "uk": "GB", "australia": "AU", "canada": "CA",
    "japan": "JP", "china": "CN", "brazil": "BR",
}

ZONE_TO_ENERGY_SOURCE = {
    "IN": "grid_india", "US": "grid_us",
    "DE": "grid_eu",    "FR": "grid_eu",
    "GB": "grid_eu",    "AU": "grid_us",
}


def get_live_intensity(country: str) -> dict:
    """Fetch live grid carbon intensity for a country."""
    zone     = COUNTRY_ZONE.get(country.lower(), "IN")
    api_key  = os.getenv("CO2SIGNAL_API_KEY", "")
    fallback = BASELINE_INTENSITY.get(
        ZONE_TO_ENERGY_SOURCE.get(zone, "grid_india"), 708
    )

    if api_key:
        for url, headers in [
            (f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}",
             {"auth-token": api_key}),
            (f"https://api.co2signal.com/v1/latest?countryCode={zone}",
             {"auth-token": api_key}),
        ]:
            try:
                resp = httpx.get(url, headers=headers, timeout=6)
                if resp.status_code == 200 and resp.text.strip():
                    data = resp.json()
                    intensity = (data.get("carbonIntensity") or
                                 data.get("data", {}).get("carbonIntensity"))
                    if intensity:
                        return {
                            "intensity_gco2_kwh": float(intensity),
                            "zone": zone, "source": "live",
                        }
            except Exception:
                continue

    return {
        "intensity_gco2_kwh": fallback,
        "zone": zone, "source": "static_fallback",
    }


def fuse_prediction(
    ml_prediction_kg: float,
    energy_source: str,
    country: str,
    kwh_per_day: float,
) -> dict:
    """
    Core fusion function.

    Formula:
        energy_fraction = (kwh * 30 * baseline_ef) / ml_prediction
        live_adjustment = live_intensity / baseline_intensity
        fused = ml_prediction * (1 - energy_fraction +
                                 energy_fraction * live_adjustment)

    This only adjusts the energy component, leaving transport/food intact.
    """
    live = get_live_intensity(country)

    baseline_ef  = BASELINE_INTENSITY.get(energy_source, 708) / 1000  # kg/kWh
    live_ef      = live["intensity_gco2_kwh"] / 1000

    # Energy contribution in the prediction
    energy_monthly_kg = kwh_per_day * 30 * baseline_ef

    # Avoid division by zero
    if ml_prediction_kg <= 0:
        return {"fused_co2_kg": 0, "adjustment_factor": 1.0,
                "live_intensity": live, "method": "no_adjustment"}

    energy_fraction = min(energy_monthly_kg / ml_prediction_kg, 0.95)

    # Adjustment factor for energy component only
    if baseline_ef > 0:
        adjustment = live_ef / baseline_ef
    else:
        adjustment = 1.0

    # Fused prediction
    fused = ml_prediction_kg * (
        (1 - energy_fraction) + (energy_fraction * adjustment)
    )
    fused = max(round(fused, 2), 0)

    return {
        "ml_baseline_kg":      round(ml_prediction_kg, 2),
        "fused_co2_kg":        fused,
        "adjustment_factor":   round(adjustment, 4),
        "energy_fraction":     round(energy_fraction, 4),
        "live_intensity_gco2": live["intensity_gco2_kwh"],
        "baseline_intensity":  BASELINE_INTENSITY.get(energy_source, 708),
        "live_source":         live["source"],
        "zone":                live["zone"],
        "method":              "hybrid_ml_live_fusion",
    }