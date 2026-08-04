"""
Ablation study for the DEPLOYED prediction formula.

This script benchmarks the exact hybrid formula that runs in production
(agent/tools.py::predict_footprint):

    xgb_pred     = XGBoost prediction, trained on a synthetically-generated
                   dataset (primary estimate; R²=0.83 on held-out test set
                   -- see research/LIMITATIONS.md for what this R² does and
                   does not validate, since the dataset's target column is
                   itself formula-generated, not measured)
    energy_delta = kwh_per_day * 30 * (live_grid_ef - baseline_grid_ef)
    final        = xgb_pred + energy_delta        (in-range inputs)
    final        = ipcc_total                     (out-of-range fallback,
                   the physical IPCC/Poore & Nemecek estimate)

An earlier production version used IPCC as primary with XGBoost clipped to
a +/-15% nudge; that scored R² < -70 on this same test set (worse than
predicting the mean) because the IPCC formula alone overestimates the real
target by roughly 2.5-3x. The "IPCC formula only" row below reproduces
that finding — keep it in the ablation table as a cautionary baseline
showing why XGBoost, not IPCC, has to be primary.

Constants below (TRANSPORT_EF, FOOD_EF, ENERGY_INTENSITY) are copied
verbatim from agent/tools.py so this script stays a faithful,
side-effect-free (no network calls, no FastAPI import) offline replica of
production. If you change the formula in agent/tools.py, update this file
to match.

KNOWN SIMPLIFICATION: the held-out test set is static historical data with
no associated "live grid intensity at request time" — there's nothing to
score the live-grid correction term against. The main ablation table below
therefore reports XGBoost's own accuracy (energy_delta=0, i.e. live
intensity == baseline intensity, which is what production also does when
the live API is unavailable). The live-grid correction's typical *effect
size* — not its accuracy — is reported separately in the sensitivity
analysis at the end, using an illustrative (not live-fetched) alternate
grid-intensity scenario.
"""

import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROCESSED_DIR = "data/processed"
ARTIFACT_DIR = "model/artifacts"

# ── Copied verbatim from agent/tools.py — keep in sync ───────────────────────
TRANSPORT_EF = {
    "car_petrol": 0.192, "car_diesel": 0.171, "car_ev": 0.053,
    "bus": 0.089, "train": 0.041, "flight_short": 0.255,
    "flight_long": 0.195, "motorcycle": 0.114, "bicycle": 0.0,
    "walking": 0.0,
}

def _load_food_ef() -> dict:
    path = f"{PROCESSED_DIR}/food_ef_real.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return {
            "beef": 27.0, "lamb": 39.2, "pork": 12.1, "chicken": 6.9,
            "fish": 6.1, "eggs": 4.8, "dairy": 3.2, "rice": 4.0,
            "vegetables": 2.0, "fruits": 1.1, "legumes": 0.9,
            "nuts": 2.5, "grains": 1.6,
        }

FOOD_EF = _load_food_ef()

ENERGY_INTENSITY = {
    "coal": 820, "natural_gas": 490, "oil": 650,
    "solar": 41, "wind": 11, "hydro": 24,
    "nuclear": 12, "grid_india": 708, "grid_us": 386,
    "grid_eu": 276,
}
ENERGY_EF = {k: v / 1000 for k, v in ENERGY_INTENSITY.items()}

# Illustrative "current, more decarbonized grid" scenario used ONLY for the
# sensitivity analysis at the end — NOT live-fetched, NOT used in the
# accuracy table. Non-grid sources (coal/solar/etc.) are point-of-generation
# and don't shift with the same-country grid mix, so they're held fixed.
ILLUSTRATIVE_CURRENT_GRID_INTENSITY = {
    "grid_india": 565, "grid_us": 350, "grid_eu": 250,
}

CAT_COLS = ["transport_type", "food_type", "energy_source"]
NUM_COLS = [
    "km_per_day", "kg_food_per_day", "kwh_per_day",
    "flights_per_year", "flight_km_total",
    "waste_co2_monthly", "clothing_co2_monthly",
    "grocery_bill_monthly", "energy_efficient",
]


def load_data():
    encoders = joblib.load(f"{PROCESSED_DIR}/encoders.pkl")
    scaler = joblib.load(f"{PROCESSED_DIR}/scaler.pkl")
    xgb_model = joblib.load(f"{ARTIFACT_DIR}/xgboost_model.pkl")
    rf_model = joblib.load(f"{ARTIFACT_DIR}/rf_model.pkl")
    try:
        xgb_model.set_params(device="cpu")
    except Exception:
        pass

    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()

    X_decoded = X_test.copy()
    for col in CAT_COLS:
        X_decoded[col] = encoders[col].inverse_transform(X_test[col].astype(int))
    X_num_orig = scaler.inverse_transform(X_test[NUM_COLS])
    for i, col in enumerate(NUM_COLS):
        X_decoded[col] = X_num_orig[:, i]

    return X_test, X_decoded, y_test, xgb_model, rf_model


def predict_xgboost(X_test, model):
    dmatrix = xgb.DMatrix(X_test)
    return model.get_booster().predict(dmatrix)


def predict_ipcc_only(X_decoded):
    """Physical formula only — no ML, static baseline grid. Matches the
    out-of-training-range fallback path in agent/tools.py."""
    preds = []
    for _, row in X_decoded.iterrows():
        transport_co2 = row["km_per_day"] * 30 * TRANSPORT_EF.get(row["transport_type"], 0.1)
        food_co2 = row["kg_food_per_day"] * 30 * FOOD_EF.get(row["food_type"], 3.0)
        energy_co2 = row["kwh_per_day"] * 30 * ENERGY_EF.get(row["energy_source"], 0.5)
        flight_co2 = (row["flight_km_total"] / 12) * TRANSPORT_EF["flight_long"]
        waste_co2 = row["waste_co2_monthly"]
        clothing_co2 = row["clothing_co2_monthly"]
        preds.append(max(
            transport_co2 + food_co2 + energy_co2 + flight_co2 + waste_co2 + clothing_co2,
            0,
        ))
    return np.array(preds)


def bootstrap_ci(y_true, y_pred, n=1000, ci=95):
    scores = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for _ in range(n):
        idx = resample(range(len(y_true)), random_state=None)
        scores.append(r2_score(y_true[idx], y_pred[idx]))
    lo = np.percentile(scores, (100 - ci) / 2)
    hi = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), lo, hi


def evaluate(name, y_true, y_pred, with_ci=False):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'-'*54}")
    print(f"  {name}")
    print(f"  MAE:  {mae:.2f} kg CO2/month")
    print(f"  RMSE: {rmse:.2f} kg CO2/month")
    print(f"  R2:   {r2:.4f}")
    ci_lo = ci_hi = None
    if with_ci:
        _, ci_lo, ci_hi = bootstrap_ci(y_true, y_pred)
        print(f"  R2 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    return {"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
             "R2": round(r2, 4), "CI_lo": ci_lo, "CI_hi": ci_hi}


def energy_delta_sensitivity(X_decoded):
    """Illustrative effect size of the live-grid correction term, using a
    plausible alternate grid-intensity scenario (NOT live-fetched, NOT used
    to score accuracy — see module docstring)."""
    deltas = []
    for _, row in X_decoded.iterrows():
        src = row["energy_source"]
        kwh = max(float(row["kwh_per_day"]), 0)
        baseline = ENERGY_INTENSITY.get(src, 500)
        current  = ILLUSTRATIVE_CURRENT_GRID_INTENSITY.get(src, baseline)
        deltas.append(kwh * 30 * (current - baseline) / 1000)
    deltas = np.array(deltas)
    grid_mask = X_decoded["energy_source"].isin(ILLUSTRATIVE_CURRENT_GRID_INTENSITY.keys())

    print("\n" + "=" * 74)
    print("LIVE-GRID CORRECTION — SENSITIVITY ANALYSIS (illustrative, not live data)")
    print("=" * 74)
    print(f"  Scenario: {ILLUSTRATIVE_CURRENT_GRID_INTENSITY}")
    print(f"  Mean delta (all rows):        {deltas.mean():+.2f} kg CO2/month")
    print(f"  Mean delta (grid users only): {deltas[grid_mask.values].mean():+.2f} kg CO2/month")
    print(f"  % of test set on grid sources: {grid_mask.sum()/len(grid_mask)*100:.1f}%")



# XGBoost hyperparameters copied from train_model.py, minus early_stopping_rounds
# / eval_set / device="cuda" -- k-fold CV holds out a different fold as the
# validation set each iteration, so there's no single fixed eval_set to early-
# stop against, and CPU-only "hist" keeps this reproducible on machines without
# a GPU. This is a deliberate, documented deviation from the exact deployed
# training recipe, not an attempt to reproduce it fold-for-fold.
CV_XGB_PARAMS = {
    "n_estimators":     600,
    "max_depth":        6,
    "learning_rate":    0.04,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "random_state":     42,
    "tree_method":      "hist",
}


def kfold_cv(k=5):
    """5-fold CV on the combined train+test set (10,000 rows), retraining
    XGBoost from scratch on each fold. The production R²=0.83 headline number
    is a single 80/20 split with no variance estimate -- this reports the
    mean and std of R²/MAE/RMSE across folds so the paper can cite a spread,
    not just a point estimate."""
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True).to_numpy()

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    print("\n" + "=" * 74)
    print(f"K-FOLD CROSS-VALIDATION (k={k}, {len(X_full)} rows, retrained per fold)")
    print("=" * 74)

    for i, (train_idx, val_idx) in enumerate(kf.split(X_full), start=1):
        model = XGBRegressor(**CV_XGB_PARAMS)
        model.fit(X_full.iloc[train_idx], y_full[train_idx])
        preds = model.predict(X_full.iloc[val_idx])

        mae = mean_absolute_error(y_full[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y_full[val_idx], preds))
        r2 = r2_score(y_full[val_idx], preds)
        fold_metrics.append({"fold": i, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"  Fold {i}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

    r2_arr = np.array([f["R2"] for f in fold_metrics])
    mae_arr = np.array([f["MAE"] for f in fold_metrics])
    rmse_arr = np.array([f["RMSE"] for f in fold_metrics])

    print("-" * 74)
    print(f"  Mean R2:   {r2_arr.mean():.4f}  (std {r2_arr.std(ddof=1):.4f})")
    print(f"  Mean MAE:  {mae_arr.mean():.2f}  (std {mae_arr.std(ddof=1):.2f})")
    print(f"  Mean RMSE: {rmse_arr.mean():.2f}  (std {rmse_arr.std(ddof=1):.2f})")
    print("\n" + "=" * 74)
    print("KEY NUMBERS FOR PAPER (k-fold CV)")
    print("=" * 74)
    print(f"  {k}-fold CV R2:  {r2_arr.mean():.4f} +/- {r2_arr.std(ddof=1):.4f}")
    print(f"  {k}-fold CV MAE: {mae_arr.mean():.2f} +/- {mae_arr.std(ddof=1):.2f} kg CO2/month")

    return fold_metrics


def run():
    print("[INFO] Loading test data...")
    X_test, X_decoded, y_test, xgb_model, rf_model = load_data()
    print(f"[INFO] Test samples: {len(y_test)}")
    print(f"[INFO] Target: mean={y_test.mean():.2f} std={y_test.std():.2f} "
          f"min={y_test.min():.2f} max={y_test.max():.2f}")

    results = []

    print("\n[RUNNING ABLATION...]")

    mean_pred = np.full(len(y_test), y_test.mean())
    results.append(evaluate("1. Mean baseline (sanity check)", y_test, mean_pred))

    ipcc_preds = predict_ipcc_only(X_decoded)
    results.append(evaluate("2. IPCC formula only (superseded primary)", y_test, ipcc_preds))

    print("\n[Running XGBoost — deployed primary predictor (with bootstrap CI)...]")
    xgb_preds = predict_xgboost(X_test, xgb_model)
    results.append(evaluate(
        "3. XGBoost (Deployed primary / Ours)", y_test, xgb_preds, with_ci=True))

    rf_preds = rf_model.predict(X_test)
    results.append(evaluate("4. RandomForest only (alternative ML baseline)", y_test, rf_preds))

    print("\n" + "=" * 74)
    print("ABLATION STUDY — FINAL TABLE")
    print("=" * 74)
    print(f"{'Model':<48} {'MAE':>7} {'RMSE':>7} {'R2':>8}  CI (95%)")
    print("-" * 74)
    for r in results:
        ci = f"[{r['CI_lo']:.4f}, {r['CI_hi']:.4f}]" if r["CI_lo"] is not None else ""
        print(f"{r['model']:<48} {r['MAE']:>7} {r['RMSE']:>7} {r['R2']:>8}  {ci}")

    ours = results[2]
    print("\n" + "=" * 74)
    print("KEY NUMBERS FOR PAPER")
    print("=" * 74)
    print(f"  Deployed primary (XGBoost) MAE:  {ours['MAE']} kg CO2/month")
    print(f"  Deployed primary (XGBoost) R2:   {ours['R2']}")
    if ours["CI_lo"] is not None:
        print(f"  R2 95% CI:                       [{ours['CI_lo']:.4f}, {ours['CI_hi']:.4f}]")
    print(f"  Improvement over IPCC-only (MAE): {results[1]['MAE'] - ours['MAE']:+.2f} kg CO2/month")
    print(f"  Improvement over IPCC-only (R2):  {ours['R2'] - results[1]['R2']:+.4f}")

    energy_delta_sensitivity(X_decoded)
    kfold_cv(k=5)


if __name__ == "__main__":
    run()
