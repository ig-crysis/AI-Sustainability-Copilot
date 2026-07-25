# Known limitations

Factual notes for the paper's Limitations section — not manuscript prose.
Re-verify against the code before citing, since this file will drift as the
system changes.

## Fixed during this pass (documented for the paper's methodology narrative)

- **Production formula was miscalibrated.** Until this pass, `predict_footprint`
  used IPCC-formula-as-primary with XGBoost clipped to a ±15% adjustment.
  Benchmarked against the held-out test set, that scored **R² = -73.8**
  (worse than predicting the mean, R²=0). Root cause: the IPCC physical
  formula overestimates the real target by ~2.5-3x on this dataset, and the
  ±15% clip couldn't correct for that. Fixed to XGBoost-primary +
  live-grid-delta correction (R²=0.83). See `backend/evaluate_ablation.py`
  for the full ablation and the historical numbers.
- **~92% of natural-language queries failed outright** (Groq
  `tool_use_failed`, HTTP 400) before this pass, because the LLM was asked
  to pre-compute derived quantities (e.g. sum device-hours × kWh rates into
  a single `kwh_per_day`) and would sometimes emit an unevaluated
  expression (e.g. `"0.05*4+0.9"`) as a tool argument, which fails
  structured-output JSON validation. Fixed by redesigning
  `predict_footprint`'s signature to accept only atomic facts (device
  hours, meal frequency, shower category) and moving all arithmetic into
  Python inside the tool. See `backend/evaluate_agent.py`.

## Open limitations

- **Single train/test split, no k-fold cross-validation.** The reported
  R²=0.83 is from one 80/20 split (`random_state=42`). No variance estimate
  across folds — only a bootstrap CI on that single split's residuals.
- **Small LLM chosen for cost/latency, not benchmarked accuracy.**
  `llama-3.1-8b-instant` was not compared against larger models on
  extraction accuracy; `backend/evaluate_agent.py` measures its accuracy in
  isolation but there's no ablation showing what a larger model would gain.
- **Live-grid-intensity correction is not independently validated.** The
  held-out test set is static historical data with no "true" live grid
  intensity at request time to score against — `evaluate_ablation.py`'s
  sensitivity analysis reports the correction term's typical *magnitude*,
  not its accuracy. When the live API is unavailable, the system silently
  falls back to a static baseline (`_fetch_live_intensity` in
  `agent/tools.py`) — this fallback rate is not currently logged/measured.
- **Lossy categorical mapping heuristics in preprocessing.**
  `data_preprocessing.py`'s `TRANSPORT_MAP`/`VEHICLE_MAP`/`HEATING_MAP`
  collapse several distinct real categories into one (e.g. `"hybrid"` and
  `"lpg"` vehicles both map to `car_petrol`; `"wood"` heating maps to
  `coal`). This is a legitimate simplification but a real source of label
  noise in training data.
- **No external validation.** No comparison against an established
  third-party carbon calculator, and no user study on whether the agent's
  suggestions are perceived as accurate or useful.
- **Reproducibility gap.** `data/raw/` and two of the three model artifact
  files (`rf_model.pkl`, `xgboost_model.pkl`) are gitignored — not
  committed. The deployed model (`best_model.pkl`/`.ubj`) is committed and
  the app runs out of the box, but reproducing the *training* pipeline from
  a fresh clone currently requires manually sourcing the raw dataset with
  no documented download link. See `backend/data/README.md` (has an open
  TODO for the exact dataset citation).
- **Dataset provenance not yet cited.** The primary training dataset
  (`backend/data/raw/Carbon_Emission.csv`) needs its exact source, author,
  and license confirmed and cited before publication — flagged as a TODO in
  `backend/data/README.md`.
- **Unused/dead artifacts found during this pass, not yet cleaned up:**
  `agent/tools.py` defines `IGES_PATH` pointing at
  `data/raw/IGES_GHG_Emissions_DB.xlsx` but never reads it, and
  `data/raw/owid-co2-data.csv` is an unused duplicate of `owid_co2.csv`.
  Harmless to the running system but worth resolving so the repo doesn't
  imply data sources that aren't actually used.
