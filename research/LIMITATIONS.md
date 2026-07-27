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
- **`TRAINING_RANGES` in-distribution guard was badly wrong for
  `kwh_per_day`.** The out-of-range fallback (route to the IPCC formula
  when an input isn't like anything the model was trained on) used a
  hardcoded bound of (0, 20.0), but the real training data's `kwh_per_day`
  ranges only 2.56–6.44 (computed directly from
  `data/processed/real_carbon_data_v2.csv`; this column is derived from
  device+shower hours in preprocessing, not a full household electricity
  bill, so it's a narrow feature by construction). A real query like "use
  15 kWh/day" passed the broken check as "in range," got a nonsensical
  XGBoost extrapolation (~119 kg), further compounded by the live-grid
  delta, landing at 68 kg/month — versus the IPCC fallback's ~492 kg, which
  is in the right ballpark against an independent estimate (~447 kg from a
  general-purpose LLM asked the same question). All five `TRAINING_RANGES`
  bounds have been recomputed from the actual processed dataset rather than
  guessed; see the comment above the constant in `agent/tools.py`. Practical
  implication: most direct household-kWh statements will now legitimately
  route to the IPCC fallback rather than XGBoost — this is correct given
  the feature's narrow training range, not a regression.

- **Rewriting the ReAct agent into a fixed 2-call pipeline (for latency,
  see the root README) silently reintroduced a `tool_use_failed` failure
  mode, undetected until `evaluate_agent.py` was re-run against the new
  code.** The rewrite grew `predict_footprint`'s docstring with more
  conditional rules; the docstring listed each categorical field's valid
  values as prose (e.g. "transport_type: car_petrol, car_diesel, ...
  bicycle, walking"). `llama-3.1-8b-instant` would sometimes misread that
  list of *allowed values* for a single field as separate *fields to
  fill in*, hallucinating extra keys (e.g. `"bicycle": 0, "car_petrol": 0`)
  alongside the real `transport_type` argument. The resulting call didn't
  match the tool's JSON schema, and Groq fell back to Llama's native
  `<function=...>` text format, which is rejected outright as
  `tool_use_failed`. Measured at **6/14 (43%) of test queries** failing
  right after the rewrite, paced at 20s between calls to rule out the
  6000 TPM rate limit as the cause (confirmed via repeated single-query
  reruns showing the same malformed-generation pattern regardless of
  load). Fixed two ways: (1) declared the five categorical arguments as
  `Literal[...]` type hints instead of prose, which compiles into an
  actual JSON schema `enum` constraint — a much stronger signal to a small
  model than a sentence to parse; (2) split the extraction call onto its
  own `temperature=0` LLM instance (previously shared at 0.2 with the
  prose-generating synthesis call) — determinism matters for a call that
  must emit structured output, not for one writing free text. Together
  these dropped the failure rate to **1/14 (7.1%)** on the same test
  cases. This is a general lesson for the paper: adding LLM-facing rules
  to a docstring is not free — each new prose constraint is a new chance
  for a small model to misparse the schema, and should be measured, not
  assumed safe.

## Open limitations

- **Residual ~7% tool-call failure rate remains, even after the fix
  above.** Confirmed via repeated single-query reruns that this is
  inherent flakiness in `llama-3.1-8b-instant`'s structured-output
  reliability, not a deterministic per-query bug — the same query
  succeeds most of the time and fails occasionally. `temperature=0`
  reduced but did not eliminate it. Not further mitigated in this pass;
  a retry-once-on-`tool_use_failed` wrapper would likely close most of
  the remaining gap cheaply.
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
- **The in-range check is a single AND-gate across all 5 features.** If
  any one of km_per_day/kg_food_per_day/kwh_per_day/flights_per_year/
  flight_km_total falls outside its training range, the *entire* prediction
  falls back to the IPCC formula, even if the other four features are
  well within range. Given `kwh_per_day`'s narrow real range (see above),
  this means many otherwise-normal queries with slightly elevated energy
  use will fall back to the physical formula rather than a per-feature
  fallback. Not fixed in this pass — a more granular per-feature confidence
  model would be a reasonable follow-up.
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
