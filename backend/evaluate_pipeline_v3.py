"""
Final ablation evaluation — corrected architecture.

Architecture insight from data:
- IPCC formula inputs (kg_food_per_day=0.3-0.5, flight_km=6774) are
  category-mapped approximations, not real quantities — IPCC overestimates 3x.
- XGBoost learned the correct target distribution directly from data (R²=0.73).
- Therefore: XGBoost is primary predictor, live grid adjusts ONLY the
  energy delta between baseline and live intensity.

Corrected hybrid formula:
  energy_delta = kwh * 30 * (live_ef - baseline_ef)
  final = xgboost_prediction + energy_delta
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.utils import resample

PROCESSED_DIR = "data/processed"
ARTIFACT_DIR  = "model/artifacts"

ENERGY_EF_BASELINE = {
    "coal": 0.820, "natural_gas": 0.490, "oil":     0.650,
    "solar": 0.041, "wind":       0.011, "hydro":   0.024,
    "nuclear": 0.012, "grid_india": 0.708,
    "grid_us": 0.386, "grid_eu":   0.276,
}
ENERGY_EF_LIVE = {
    "coal": 0.820, "natural_gas": 0.490, "oil":     0.650,
    "solar": 0.041, "wind":       0.011, "hydro":   0.024,
    "nuclear": 0.012, "grid_india": 0.565,
    "grid_us": 0.350, "grid_eu":   0.250,
}


def load_data():
    encoders = joblib.load(f"{PROCESSED_DIR}/encoders.pkl")
    scaler   = joblib.load(f"{PROCESSED_DIR}/scaler.pkl")
    model    = joblib.load(f"{ARTIFACT_DIR}/best_model.pkl")
    try:
        model.set_params(device="cpu")
    except Exception:
        pass

    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()

    cat_cols = ["transport_type", "food_type", "energy_source"]
    num_cols = ["km_per_day", "kg_food_per_day", "kwh_per_day",
                "flights_per_year", "flight_km_total"]

    X_decoded = X_test.copy()
    for col in cat_cols:
        X_decoded[col] = encoders[col].inverse_transform(
            X_test[col].astype(int)
        )
    X_num_orig = scaler.inverse_transform(X_test[num_cols])
    for i, col in enumerate(num_cols):
        X_decoded[col] = X_num_orig[:, i]

    return X_test, X_decoded, y_test, model


def predict_xgboost(X_test, model):
    """XGBoost primary prediction."""
    dmatrix = xgb.DMatrix(X_test)
    return model.get_booster().predict(dmatrix)


def predict_xgb_live_grid(X_test, X_decoded, model):
    """
    Corrected hybrid:
    XGBoost prediction + live grid energy delta.

    The XGBoost model was trained with baseline grid intensities.
    When the live grid intensity differs from baseline, the energy
    component of the footprint changes. We apply this delta additively:

      energy_delta_kg = kwh_per_day * 30 * (live_ef - baseline_ef)
      final = xgb_prediction + energy_delta_kg

    This is mathematically sound because:
    - XGBoost already accounts for energy at baseline intensity
    - The delta captures only the real-time grid variation
    - No unit mismatch — both terms are in kg CO2/month
    """
    xgb_preds = predict_xgboost(X_test, model)
    deltas    = []

    for _, row in X_decoded.iterrows():
        src          = row["energy_source"]
        kwh          = max(float(row["kwh_per_day"]), 0)
        baseline_ef  = ENERGY_EF_BASELINE.get(src, 0.5)
        live_ef      = ENERGY_EF_LIVE.get(src, baseline_ef)
        delta        = kwh * 30 * (live_ef - baseline_ef)
        deltas.append(delta)

    deltas      = np.array(deltas)
    final_preds = np.maximum(xgb_preds + deltas, 0)

    print(f"  [INFO] XGBoost mean:          {xgb_preds.mean():.2f} kg/month")
    print(f"  [INFO] Mean energy delta:     {deltas.mean():.2f} kg/month")
    print(f"  [INFO] Final hybrid mean:     {final_preds.mean():.2f} kg/month")
    print(f"  [INFO] Grid sources in test:  "
          f"{pd.Series([X_decoded.iloc[i]['energy_source'] for i in range(len(X_decoded))]).value_counts().to_dict()}")

    return final_preds


def bootstrap_ci(y_true, y_pred, n=1000, ci=95):
    scores  = []
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    for _ in range(n):
        idx = resample(range(len(y_true)), random_state=None)
        scores.append(r2_score(y_true[idx], y_pred[idx]))
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), lo, hi


def evaluate(name, y_true, y_pred, with_ci=False):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*54}")
    print(f"  {name}")
    print(f"  MAE:  {mae:.2f} kg CO₂/month")
    print(f"  RMSE: {rmse:.2f} kg CO₂/month")
    print(f"  R²:   {r2:.4f}")
    ci_lo = ci_hi = None
    if with_ci:
        _, ci_lo, ci_hi = bootstrap_ci(y_true, y_pred)
        print(f"  R² 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    return {"model": name, "MAE": round(mae,2), "RMSE": round(rmse,2),
            "R2": round(r2,4), "CI_lo": ci_lo, "CI_hi": ci_hi}


def run():
    print("[INFO] Loading test data...")
    X_test, X_decoded, y_test, model = load_data()
    print(f"[INFO] Test samples:  {len(y_test)}")
    print(f"[INFO] Target:  mean={y_test.mean():.2f}  "
          f"std={y_test.std():.2f}  "
          f"min={y_test.min():.2f}  max={y_test.max():.2f}")

    results = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n[RUNNING ABLATION...]")

    xgb_preds = predict_xgboost(X_test, model)
    results.append(evaluate(
        "XGBoost-only (ML baseline)",
        y_test, xgb_preds))

    # Naive mean baseline (sanity check)
    mean_pred = np.full(len(y_test), y_test.mean())
    results.append(evaluate(
        "Mean baseline (predict mean for all)",
        y_test, mean_pred))

    # ── Our system ────────────────────────────────────────────────────────────
    print("\n[Running XGBoost + Live Grid (with CI — ~30s)...]")
    hybrid_preds = predict_xgb_live_grid(X_test, X_decoded, model)
    results.append(evaluate(
        "XGBoost + Live Grid Adjustment (Ours)",
        y_test, hybrid_preds, with_ci=True))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ABLATION STUDY — FINAL TABLE")
    print("="*70)
    print(f"{'Model':<42} {'MAE':>7} {'RMSE':>7} {'R²':>8}  CI (95%)")
    print("-"*70)
    for r in results:
        ci = ""
        if r["CI_lo"] is not None:
            ci = f"[{r['CI_lo']:.4f}, {r['CI_hi']:.4f}]"
        print(f"{r['model']:<42} {r['MAE']:>7} {r['RMSE']:>7} "
              f"{r['R2']:>8}  {ci}")

    # ── Key stats for paper ────────────────────────────────────────────────────
    xgb_r2    = results[0]["R2"]
    hybrid_r2 = results[2]["R2"]
    xgb_mae   = results[0]["MAE"]
    hybrid_mae = results[2]["MAE"]

    print("\n" + "="*70)
    print("KEY NUMBERS FOR IEEE PAPER")
    print("="*70)
    print(f"  XGBoost MAE:                  {xgb_mae} kg CO₂/month")
    print(f"  XGBoost R²:                   {xgb_r2}")
    print(f"  XGBoost + Live Grid MAE:      {hybrid_mae} kg CO₂/month")
    print(f"  XGBoost + Live Grid R²:       {hybrid_r2}")
    if results[2]["CI_lo"]:
        print(f"  R² 95% CI:                    "
              f"[{results[2]['CI_lo']:.4f}, {results[2]['CI_hi']:.4f}]")
    print(f"  Live grid improvement (MAE):  "
          f"{xgb_mae - hybrid_mae:+.2f} kg CO₂/month")
    print(f"  Live grid improvement (R²):   "
          f"{hybrid_r2 - xgb_r2:+.4f}")

    # Energy delta stats
    deltas = []
    for _, row in X_decoded.iterrows():
        src = row["energy_source"]
        kwh = max(float(row["kwh_per_day"]), 0)
        delta = kwh * 30 * (ENERGY_EF_LIVE.get(src, 0.5) -
                            ENERGY_EF_BASELINE.get(src, 0.5))
        deltas.append(delta)
    deltas = np.array(deltas)
    grid_mask = X_decoded["energy_source"].isin(
        ["grid_india", "grid_us", "grid_eu"])
    print(f"\n  Live grid delta stats (all):       "
          f"mean={deltas.mean():.2f}, std={np.std(deltas):.2f}")
    print(f"  Live grid delta (grid users only): "
          f"mean={deltas[grid_mask.values].mean():.2f} kg CO₂/month")
    print(f"  % of test set using grid sources:  "
          f"{grid_mask.sum()/len(grid_mask)*100:.1f}%")


if __name__ == "__main__":
    run()