import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Real food emission factors from Poore & Nemecek ──────────────────────────
FOOD_EF_PKG = os.path.join(PROCESSED_DIR, "food_ef_real.pkl")

def load_food_ef():
    if os.path.exists(FOOD_EF_PKG):
        return joblib.load(FOOD_EF_PKG)
    return {"beef":27.0,"lamb":39.2,"pork":12.1,"chicken":6.9,
            "fish":6.1,"eggs":4.8,"dairy":3.2,"rice":4.0,
            "vegetables":2.0,"fruits":1.1,"legumes":0.9,"nuts":2.5}

# ── Meal serving size by diet type (kg per meal, realistic) ──────────────────
DIET_TO_FOOD_EXACT = {
    "vegan":       ("vegetables", 0.30, 3),   # (food_type, kg/meal, meals/day)
    "vegetarian":  ("dairy",      0.25, 3),
    "pescatarian": ("fish",       0.15, 1),   # fish once/day, rest vegetarian
    "omnivore":    ("chicken",    0.15, 1),   # meat once/day
}

# ── Shower frequency → water heating kWh/day ────────────────────────────────
SHOWER_KWH = {
    "daily":             0.90,
    "less frequently":   0.45,
    "more frequently":   1.35,
    "twice a day":       1.80,
}

# ── Transport mapping with real km from dataset ──────────────────────────────
TRANSPORT_MAP = {
    "public":       "bus",
    "walk/bicycle": "bicycle",
    "private":      "car_petrol",
}
VEHICLE_MAP = {
    "petrol":   "car_petrol",
    "diesel":   "car_diesel",
    "electric": "car_ev",
    "hybrid":   "car_petrol",
    "lpg":      "car_petrol",
}

# ── Heating energy → source ──────────────────────────────────────────────────
HEATING_MAP = {
    "coal":        "coal",
    "natural gas": "natural_gas",
    "wood":        "coal",
    "electricity": "grid_eu",
}

# ── Flight frequency → annual flights + km ──────────────────────────────────
FLIGHT_MAP = {
    "never":           (0,   0),
    "rarely":          (1,   2000),
    "frequently":      (4,   8000),
    "very frequently": (8,  16000),
}

# ── Waste bag → kg waste/week → approx CO2 ──────────────────────────────────
WASTE_CO2_WEEKLY = {
    "small":       0.5,
    "medium":      1.0,
    "large":       2.0,
    "extra large": 3.5,
}

# ── Clothing → manufacturing CO2 (kg/item) ──────────────────────────────────
CLOTHING_CO2_PER_ITEM = 10.0  # avg fast fashion item lifecycle

# ── Device kWh/hour ─────────────────────────────────────────────────────────
TV_PC_KWH_HR   = 0.1   # desktop/tv average
INTERNET_KWH_HR = 0.01  # modem/router


def map_row(row, food_ef):
    """Map one raw dataset row to structured features + target."""
    try:
        # ── Transport ────────────────────────────────────────────────────────
        transport_raw = str(row.get("Transport", "")).lower().strip()
        vehicle_raw   = str(row.get("Vehicle Type", "")).lower().strip()
        transport_type = TRANSPORT_MAP.get(transport_raw, "car_petrol")
        if transport_raw == "private" and vehicle_raw in VEHICLE_MAP:
            transport_type = VEHICLE_MAP[vehicle_raw]

        # Use actual km from dataset — this is the key fix
        monthly_km = float(row.get("Vehicle Monthly Distance Km", 0) or 0)
        km_per_day = monthly_km / 30.0

        # ── Food — use real serving sizes ────────────────────────────────────
        diet      = str(row.get("Diet", "omnivore")).lower().strip()
        food_type, kg_per_meal, meals_per_day = DIET_TO_FOOD_EXACT.get(
            diet, ("chicken", 0.15, 1)
        )
        kg_food_per_day = kg_per_meal * meals_per_day

        # ── Energy — heating + appliances ───────────────────────────────────
        heating    = str(row.get("Heating Energy Source", "electricity")).lower().strip()
        energy_src = HEATING_MAP.get(heating, "grid_eu")

        shower_kwh = SHOWER_KWH.get(
            str(row.get("How Often Shower", "daily")).lower().strip(), 0.9
        )
        tv_hrs     = float(row.get("How Long TV PC Daily Hour", 4) or 4)
        internet_hrs = float(row.get("How Long Internet Daily Hour", 4) or 4)
        device_kwh = tv_hrs * TV_PC_KWH_HR + internet_hrs * INTERNET_KWH_HR
        kwh_per_day = round(shower_kwh + device_kwh + 2.0, 2)  # 2kWh base

        # ── Flights ──────────────────────────────────────────────────────────
        freq_raw       = str(row.get("Frequency of Traveling by Air", "never")).lower().strip()
        flights_yr, flight_km = FLIGHT_MAP.get(freq_raw, (0, 0))

        # ── Lifestyle features (XGBoost extras — no IPCC factor) ─────────────
        waste_size   = str(row.get("Waste Bag Size", "medium")).lower().strip()
        waste_count  = float(row.get("Waste Bag Weekly Count", 1) or 1)
        waste_co2_mo = WASTE_CO2_WEEKLY.get(waste_size, 1.0) * waste_count * 4.33

        new_clothes  = float(row.get("How Many New Clothes Monthly", 2) or 2)
        clothing_co2_mo = new_clothes * CLOTHING_CO2_PER_ITEM / 12  # annualized

        grocery_bill = float(row.get("Monthly Grocery Bill", 200) or 200)

        energy_efficient = 1 if str(row.get("Energy efficiency","No")).strip()=="Yes" else 0

        # ── Target ───────────────────────────────────────────────────────────
        annual_co2  = float(row["CarbonEmission"])
        monthly_co2 = round(annual_co2 / 12, 2)

        return {
            # Primary IPCC features
            "transport_type":    transport_type,
            "km_per_day":        round(km_per_day, 2),
            "food_type":         food_type,
            "kg_food_per_day":   round(kg_food_per_day, 4),
            "energy_source":     energy_src,
            "kwh_per_day":       kwh_per_day,
            "flights_per_year":  flights_yr,
            "flight_km_total":   float(flight_km),
            # Lifestyle features (XGBoost only)
            "waste_co2_monthly": round(waste_co2_mo, 2),
            "clothing_co2_monthly": round(clothing_co2_mo, 2),
            "grocery_bill_monthly": round(grocery_bill, 2),
            "energy_efficient":  energy_efficient,
            # Target
            "monthly_co2_kg":    monthly_co2,
        }
    except Exception as e:
        return None


def run():
    path = os.path.join(RAW_DIR, "Carbon_Emission.csv")
    df   = pd.read_csv(path)
    print(f"[INFO] Raw dataset: {df.shape}")
    print(f"[INFO] Target stats: mean={df['CarbonEmission'].mean()/12:.1f} "
          f"std={df['CarbonEmission'].std()/12:.1f} kg/month")

    food_ef = load_food_ef()
    records = []
    skipped = 0
    for _, row in df.iterrows():
        mapped = map_row(row, food_ef)
        if mapped:
            records.append(mapped)
        else:
            skipped += 1

    result = pd.DataFrame(records)
    print(f"[INFO] Mapped: {result.shape}, skipped: {skipped}")
    print(f"[INFO] km_per_day: mean={result['km_per_day'].mean():.2f} "
          f"(was hardcoded ~20)")
    print(f"[INFO] kg_food_per_day: mean={result['kg_food_per_day'].mean():.4f} "
          f"(was hardcoded 0.4)")
    print(f"[INFO] waste_co2_monthly: mean={result['waste_co2_monthly'].mean():.2f}")
    print(f"[INFO] clothing_co2_monthly: mean={result['clothing_co2_monthly'].mean():.2f}")

    # Save full mapped dataset
    result.to_csv(f"{PROCESSED_DIR}/real_carbon_data_v2.csv", index=False)

    # ── Encode and split ──────────────────────────────────────────────────────
    cat_cols = ["transport_type", "food_type", "energy_source"]
    num_cols = ["km_per_day", "kg_food_per_day", "kwh_per_day",
                "flights_per_year", "flight_km_total",
                "waste_co2_monthly", "clothing_co2_monthly",
                "grocery_bill_monthly", "energy_efficient"]
    target   = "monthly_co2_kg"

    df_clean = result.dropna(subset=[target]).copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    X = df_clean[cat_cols + num_cols]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv",   index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv",   index=False)
    joblib.dump(encoders, f"{PROCESSED_DIR}/encoders.pkl")
    joblib.dump(scaler,   f"{PROCESSED_DIR}/scaler.pkl")

    print(f"\n[DONE] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[DONE] Target: mean={y.mean():.2f}, std={y.std():.2f} kg/month")
    print(f"[DONE] Features: {list(X.columns)}")
    print(f"[DONE] Files saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    run()