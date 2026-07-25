# Data provenance

## Pipeline (what actually feeds the deployed model)

```
data/raw/Carbon_Emission.csv          (raw survey dataset, ground-truth CarbonEmission column)
        │
        ▼  data_preprocessing.py (see map_row())
data/processed/real_carbon_data_v2.csv   (mapped features + monthly_co2_kg target)
        │
        ▼  train/test split (80/20, random_state=42)
data/processed/{X,y}_{train,test}.csv
data/processed/encoders.pkl, scaler.pkl
        │
        ▼  train_model.py
model/artifacts/xgboost_model.pkl, rf_model.pkl, best_model.pkl (+ .ubj native copy)
```

The deployed model (`agent/tools.py::predict_footprint`) is XGBoost, trained
directly on the real `CarbonEmission` target from this dataset — it is not
fit to any formula-generated target. See `backend/evaluate_ablation.py` for
held-out test-set performance (R²≈0.83).

## File-by-file status

| File | Status | Notes |
|---|---|---|
| `data/raw/Carbon_Emission.csv` | **Active — primary training data** | 10,000-row individual/household carbon footprint survey dataset. **TODO(user): fill in the exact source URL, author, and license** — this repo's `data/raw/` is gitignored (not committed), so anyone reproducing this pipeline needs the original download link. If this is a Kaggle dataset, cite it by its Kaggle listing (dataset name + author + URL) in the root `README.md` citations section once confirmed. |
| `data/raw/Food_Product_Emissions.csv` | Referenced by `food_ef_real.pkl` generation | Confirm and cite source (looks derived from Poore & Nemecek 2018, already cited elsewhere in the code, but verify this specific file matches that source before citing it as such in a paper). |
| `data/raw/owid_co2.csv` | **Active** | Read directly by `precompute_owid.py` to build `data/processed/owid_baselines.pkl` (regional per-capita baselines used by `get_regional_baseline`). Source: Our World in Data CO2 dataset. |
| `data/raw/owid-co2-data.csv` | Unused duplicate | Same OWID dataset, different filename, not read by any script (`precompute_owid.py` hardcodes `owid_co2.csv`). Delete to avoid the "which one is real" ambiguity, or note explicitly if it's kept as a raw-download reference copy. |
| `data/raw/IGES_GHG_Emissions_DB.xlsx` | Unreferenced in current code | `agent/tools.py` defines `IGES_PATH` pointing at this file but never reads it — confirmed via grep, only the constant definition exists, no `pd.read_excel`/`open` call anywhere. Either wire it in or remove both the file and the dead constant. |
| ~~`data/raw/synthetic_carbon_data.csv`~~ | **Removed** | Was not imported anywhere in the codebase (confirmed via grep before deletion). Same 9-column schema as `real_carbon_data.csv` (v1, below) — likely an early synthetic placeholder used before the real Kaggle dataset was obtained. If you have a specific reason to keep it (e.g. it appears in an earlier paper draft or notebook), it's recoverable from git history (`git log --all --full-history -- backend/data/raw/synthetic_carbon_data.csv`) — but it never reached this commit since `data/raw/` is gitignored. |
| `data/processed/real_carbon_data.csv` | **Stale (v1)** | 9 columns, no lifestyle features (waste/clothing/grocery/energy_efficient). Superseded by `real_carbon_data_v2.csv`, which is what `train_model.py` actually trains on (via the `X_train`/`X_test` splits). Consider deleting v1 or clearly marking it archival to avoid ambiguity about which dataset produced the deployed model. |
| `data/processed/real_carbon_data_v2.csv` | **Active** | Output of `data_preprocessing.py::run()`. This is the dataset the deployed model was trained on. |

## Reproducibility gap (read before citing "reproducible" in a paper)

`backend/data/raw/` and `backend/model/artifacts/{rf,xgboost}_model.pkl` are
**gitignored** — not committed. `model/artifacts/best_model.pkl` and
`best_model.ubj` (the files actually loaded at inference time) *are*
committed, so the deployed app works out of the box. But re-running
`data_preprocessing.py` → `train_model.py` → `evaluate_ablation.py` from a
fresh clone currently requires manually sourcing the raw CSVs — there are no
download instructions anywhere in the repo. Fix this by adding the exact
source link(s) to the TODOs above and to the root `README.md` before
submission, so a reviewer can actually reproduce the training run.

## Preprocessing heuristics (for the paper's Methods/Limitations section)

`data_preprocessing.py::map_row()` converts categorical survey answers into
the model's input features using several many-to-one mapping tables
(`TRANSPORT_MAP`, `VEHICLE_MAP`, `HEATING_MAP`, `FLIGHT_MAP`). These are
lossy approximations — e.g. `"hybrid"` and `"lpg"` vehicles are both mapped
to `car_petrol`, and `"wood"` heating is mapped to `coal`. This is a
legitimate modeling choice but should be stated explicitly as a limitation
(see `research/LIMITATIONS.md`), not left implicit in the source code.
