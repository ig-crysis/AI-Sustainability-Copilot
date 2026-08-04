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

- ~~**Residual ~7% tool-call failure rate remains, even after the fix
  above.**~~ **RESOLVED (2026-07-28, `agent/carbon_agent.py::_run_extraction`,
  `_recover_native_function_call`).** The "inherent flakiness" framing above
  turned out to be wrong for at least one confirmed case: the residual
  1/14 failure (`direct_food_quantity_stated` in `evaluate_agent.py`) was
  re-run 4 times total across this pass (2 full harness runs + 2 isolated
  single-query repro calls) and failed **all 4 times**, identically, at
  `temperature=0` — i.e. deterministic for this query, not random. A retry
  with the same messages can never fix a deterministic failure, so a
  retry-only wrapper was insufficient on its own (confirmed empirically:
  wrapping the extraction call in a retry-once-on-`tool_use_failed`
  handler, verified correct via a mocked unit test, still left this exact
  case failing in two separate live-API eval runs).

  Inspecting the raw Groq error body (`e.body["error"]["failed_generation"]`)
  revealed the actual root cause: the model's extraction was **already
  fully correct** — `<function=predict_footprint>{"transport_type": "bus",
  "km_per_day": 20, "total_kg_food_per_day": 0.3, "energy_source": "wind",
  ...}` — but emitted in Llama's native `<function=name>{...}` text format
  instead of Groq's expected structured tool-call schema, and Groq rejects
  the *entire* response outright based on that format mismatch alone,
  regardless of whether the content is right. Fixed by parsing this
  rejected text as a recovery path (`_recover_native_function_call`): on
  `tool_use_failed`, extract and JSON-parse the `<function=...>{...}`
  payload from `failed_generation` before falling back to a same-request
  retry (which is kept as a second layer for any genuinely stochastic
  failures, since it costs nothing when recovery succeeds on the first
  try). Verified by direct repro: the previously deterministic 4/4-failing
  query now succeeds on every call by recovering the correct extraction.
  Re-running the full `evaluate_agent.py` suite once more after this fix
  gave **0/14 `tool_use_failed` errors** (the one agent error in that run
  was an unrelated `RateLimitError` 429 — the already-documented 6000 TPM
  Groq cap, triggered by three eval runs in quick succession — not a
  tool-call formatting failure). General lesson for the paper: a
  `tool_use_failed` rejection is a claim about *response format*, not
  necessarily about *extraction correctness* — worth checking
  `failed_generation` before assuming the underlying reasoning was wrong.
- ~~**Single train/test split, no k-fold cross-validation.**~~ **RESOLVED
  (2026-07-28, `backend/evaluate_ablation.py::kfold_cv`).** 5-fold CV on the
  combined 10,000-row train+test set (XGBoost retrained from scratch per
  fold, same hyperparameters as `train_model.py` minus early stopping/GPU —
  there's no single fixed validation set to early-stop against when the held-
  out fold changes each iteration) gives **R² = 0.8217 ± 0.0044** and
  **MAE = 25.92 ± 0.24 kg CO2/month** across folds — tight variance,
  consistent with the original single-split R²=0.83 and its bootstrap CI
  [0.8135, 0.8439]. The single-split number was not a lucky split. This
  addresses the CV gap; it does **not** address the separate, larger issue
  below (synthetic target, no external validation).
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
- **Measured (2026-07-28, `backend/evaluate_routing.py`): XGBoost — the
  "primary, R²=0.83" model — only fires on ~30% of realistic queries by the
  numeric-range gate alone, and disagrees sharply with the IPCC fallback on
  the cases where both are computable.** The in-range check is a single
  AND-gate across all 5 numeric features
  (km_per_day/kg_food_per_day/kwh_per_day/flights_per_year/flight_km_total)
  — if any one falls outside its training range, the *entire* prediction
  falls back to the IPCC formula, even if the other four are well within
  range. Over 3,000 randomly-sampled realistic scenarios (not dataset rows —
  see the script for the sampling procedure): **895/3000 (29.8%) passed the
  numeric-range gate.** Of the numeric fallback cases, `kwh_per_day` was the
  single biggest blocker (56.2%), with `flight_km_total` (33.9%) and
  `kg_food_per_day` (30.0%) comparably large contributors. Root cause for
  those last two, confirmed by reading `data_preprocessing.py`: they aren't
  narrow because of sampling — they're *derived from small categorical
  lookup tables* in the raw dataset (`FLIGHT_MAP` has exactly 4 frequency
  buckets → 4 possible `flight_km_total` values; `DIET_TO_FOOD_EXACT` has 4
  diet types → effectively 3 distinct `kg_food_per_day` values). A
  continuous, realistic `flight_km_total` or `kg_food_per_day` was never
  something the model was trained to interpolate over, which is why simply
  widening `TRAINING_RANGES` for these fields would not be a safe fix (see
  the kwh_per_day incident above, where widening a range let a genuinely
  out-of-distribution input reach the model and produce a nonsensical
  prediction) — not attempted for this reason.

  **FIXED — second, larger routing bug found and closed this pass
  (2026-07-28, `agent/tools.py::_check_categorical_in_vocab`,
  `backend/evaluate_routing.py`): the numeric-range gate above was never the
  full story, because `transport_type`/`food_type`/`energy_source` had no
  vocabulary check at all.** `predict_footprint`'s `Literal` type hints
  expose the full IPCC/Poore & Nemecek category sets — 10 transport types,
  13 food types, 10 energy sources — so the extraction LLM can name whatever
  the user actually said. But the raw training dataset only contains a
  handful of category labels per column (confirmed directly from
  `data/processed/encoders.pkl`): **`transport_type` ∈ {bicycle, bus,
  car_diesel, car_ev, car_petrol} (5/10), `food_type` ∈ {chicken, dairy,
  fish, vegetables} (4/13), `energy_source` ∈ {coal, grid_eu, natural_gas}
  (3/10)**. Critically, **`grid_india` — `predict_footprint`'s own default
  energy_source — was never in the trained vocabulary.** Before this fix,
  any category outside that list hit `except ValueError: df_pred[col] = 0`
  inside `predict_footprint`, which silently re-mapped it to whichever class
  happens to be encoded as 0 (e.g. any untrained energy_source silently
  became "coal") and *still returned an XGBoost prediction* with
  `ml_in_training_range: True` — a real, silently-wrong-answer bug in
  production, not merely an accuracy gap. Fixed by folding categorical
  vocabulary membership into the same in/out-of-training-distribution gate
  as the 5 numeric ranges, so an unrecognized category now correctly
  triggers the IPCC fallback instead of a miscoded ML prediction.

  **Re-measuring after the fix reveals the true picture is far more
  extreme than the 29.8% numeric-only figure suggested: only 25/3000
  (0.8%) of realistic scenarios pass BOTH the numeric range gate and the
  categorical vocabulary gate.** `energy_source` alone fails 70.5% of all
  3,000 scenarios (a direct consequence of `grid_india`/`grid_us` not being
  trained categories), `food_type` fails 69.9%, `transport_type` fails
  51.1%. **In production terms: XGBoost, the model this project reports
  R²=0.83 for, correctly and safely fires on well under 1% of realistic
  natural-language queries; the IPCC formula is the de facto primary
  predictor for the deployed system, not XGBoost.** This is a stronger and
  more important claim than the original 29.8% finding, and arguably the
  single most consequential limitation in the whole project — more so than
  the synthetic-data provenance issue, because it means the R²=0.83 model
  essentially never actually runs on real traffic.

  Among the 25 (now correctly gated) in-range scenarios, XGBoost disagreed
  with the IPCC estimate by a mean of 581.53 kg CO2/month (69.3% relative;
  XGBoost averaged 157 kg/month vs. IPCC's 728 kg/month), 84% of those
  scenarios differing by more than 50%. n=25 is small — this sub-statistic
  should be read as indicative, not precise. This is not a ground-truth
  comparison — both methods estimate the same synthetic-target-fit quantity
  differently — but it quantifies how much the routing gate's decision
  swings the number a user actually sees, with no uncertainty indication
  surfaced to them either way. **Not fixed in this pass: expanding the
  trained categorical vocabulary would require retraining on data the raw
  Kaggle dataset doesn't have (it only contains 4 diet types, 3 heating
  sources, etc. to begin with) — this is a training-data coverage
  limitation, not a routing-logic bug, and is out of scope without a richer
  dataset.** Surfacing the disagreement/uncertainty to the user, or
  reframing the system's primary predictor as the IPCC formula (with
  XGBoost as a rare high-confidence override) rather than the reverse,
  would be reasonable follow-ups.
- **No external validation.** No comparison against an established
  third-party carbon calculator, and no user study on whether the agent's
  suggestions are perceived as accurate or useful.
- **Reproducibility gap.** `data/raw/` and two of the three model artifact
  files (`rf_model.pkl`, `xgboost_model.pkl`) are gitignored — not
  committed. The deployed model (`best_model.pkl`/`.ubj`) is committed and
  the app runs out of the box, but reproducing the *training* pipeline from
  a fresh clone currently requires manually sourcing the raw dataset. See
  `backend/data/README.md` for the confirmed download link/citation.
- **The primary training dataset's target column is synthetically
  generated, not measured real-world data — this is the most important
  limitation for the paper's Methods section, more so than a citation gap.**
  Confirmed directly on the dataset's Kaggle listing (2026-07-28):
  "Individual Carbon Footprint Calculation" by Mesut Duman and 4
  collaborators, https://www.kaggle.com/datasets/dumanmesut/individual-carbon-footprint-calculation,
  License: CC0: Public Domain. The listing itself states: *"The data has
  been synthetically generated, calculated based on weightings from
  various studies and sites that currently compute the dependent variable,
  carbon emissions, attempting to maintain values close to reality."* This
  directly contradicts prior wording throughout this codebase and docs
  (the filename `real_carbon_data_v2.csv`, and phrases like "trained
  directly on real survey data" previously in `README.md`,
  `backend/data/README.md`, `agent/tools.py`'s docstring/comments/API
  response text, `agent/fusion.py`, and `evaluate_ablation.py`) — all now
  corrected to describe the dataset as synthetic. **Practical implication
  for how to interpret the reported R²=0.83**: it measures how well
  XGBoost recovers the dataset creators' unpublished weighting
  formula/heuristic from the input features, not real-world predictive
  accuracy against independently-measured footprints. The model could be
  a very accurate emulator of that formula and still diverge meaningfully
  from real emissions if the formula's weightings are themselves
  inaccurate approximations (their own listing hedges this: "attempting
  to maintain values close to reality," not validated against real
  measurements). This should be stated plainly and early in the paper's
  Methods/Data section, not left implicit — a reviewer who discovers the
  Kaggle listing's synthetic-data disclosure independently, after reading
  a paper that implies real survey data, would reasonably read that as a
  methodological misrepresentation.
- **Unused/dead artifacts found during this pass, not yet cleaned up:**
  `agent/tools.py` defines `IGES_PATH` pointing at
  `data/raw/IGES_GHG_Emissions_DB.xlsx` but never reads it,
  `data/raw/owid-co2-data.csv` is an unused duplicate of `owid_co2.csv`,
  and `agent/fusion.py` (a "Hybrid Fusion Layer" module) is not imported
  anywhere in the running application — `main.py` and `agent/tools.py`
  implement the actual fusion logic directly. Harmless to the running
  system but worth resolving so the repo doesn't imply data sources or
  modules that aren't actually used.
