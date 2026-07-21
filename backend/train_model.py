import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
import shutil

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

PROCESSED_DIR = "data/processed"
ARTIFACT_DIR  = "model/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_experiment("carbon-footprint-real-data")


def load_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    print(f"\n── {name} ──")
    print(f"   MAE:  {mae:.2f} kg CO₂/month")
    print(f"   RMSE: {rmse:.2f} kg CO₂/month")
    print(f"   R²:   {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run():
    print("[INFO] Loading real dataset...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Target range: {y_train.min():.1f} – {y_train.max():.1f} kg CO2/month")

    results     = {}
    filename_map = {}

    # ── XGBoost with early stopping ──────────────────────────────────────────
    xgb_params = {
        "n_estimators":     600,
        "max_depth":        6,
        "learning_rate":    0.04,
        "subsample":        0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "random_state":     42,
        "device":           "cuda",
        "tree_method":      "hist",
        "early_stopping_rounds": 50,
    }
    with mlflow.start_run(run_name="XGBoost-RealData"):
        model_xgb = XGBRegressor(**xgb_params)
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100,
        )
        m = evaluate(model_xgb, X_test, y_test, "XGBoost-RealData")
        mlflow.log_params(xgb_params)
        mlflow.log_metrics(m)
        joblib.dump(model_xgb, f"{ARTIFACT_DIR}/xgboost_model.pkl")
        results["xgboost"]      = m
        filename_map["xgboost"] = "xgboost_model.pkl"

    # ── RandomForest baseline ────────────────────────────────────────────────
    rf_params = {
        "n_estimators": 300,
        "max_depth":    12,
        "random_state": 42,
        "n_jobs":       -1,
    }
    with mlflow.start_run(run_name="RandomForest-RealData"):
        model_rf = RandomForestRegressor(**rf_params)
        model_rf.fit(X_train, y_train)
        m = evaluate(model_rf, X_test, y_test, "RandomForest-RealData")
        mlflow.log_params(rf_params)
        mlflow.log_metrics(m)
        joblib.dump(model_rf, f"{ARTIFACT_DIR}/rf_model.pkl")
        results["rf"]      = m
        filename_map["rf"] = "rf_model.pkl"

    # ── Save best model ──────────────────────────────────────────────────────
    best = min(results, key=lambda k: results[k]["mae"])
    src  = os.path.join(ARTIFACT_DIR, filename_map[best])
    dst  = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    shutil.copy(src, dst)

    print(f"\n[BEST] {best} — MAE={results[best]['mae']:.2f}, R²={results[best]['r2']:.4f}")
    print(f"[SAVED] best_model.pkl ← {filename_map[best]}")
    print("[INFO] Run: mlflow ui  to see experiment comparison")


if __name__ == "__main__":
    run()