> **Editor's note (2026-07-28):** This revision replaces the earlier draft entirely.
> The previous version described a design that predates this project's debugging
> history: an IPCC-primary/XGBoost-adjustment pipeline (since measured at
> R²=-73.8 and replaced), a LangGraph ReAct agent on Llama 3.3 70B (since
> replaced with a fixed 2-call pipeline on llama-3.1-8b-instant for latency and
> Groq rate-limit reasons), and a "10,000 real lifestyle records" dataset
> characterization that contradicts the dataset's own Kaggle listing (it is
> synthetically generated). Every number below is reproducible from this repo:
> `backend/evaluate_ablation.py` (accuracy/CV), `backend/evaluate_agent.py`
> (extraction reliability), `backend/evaluate_routing.py` (routing gate). See
> `research/LIMITATIONS.md` for the full evidence trail. Where the original
> draft made a claim this pass could not verify against the code (the
> Duman–Ozcan 0.68–0.74 benchmark, the 100%-vs-68% threshold-accuracy figure,
> the IGES GHG dataset, the 2.1s latency figure, the live-grid-fusion accuracy
> gain), it has been removed rather than carried forward unverified.

# AI Sustainability Copilot: An LLM-Orchestrated Hybrid ML Framework for Personalized Carbon Footprint Estimation

**Ritika Kanwar, Anant Bhatnagar**
Dept. of SCORE–MCA, VIT University, Vellore, Tamil Nadu, India

## Abstract

We present the AI Sustainability Copilot, a hybrid system combining a trained
XGBoost regression model, an LLM-based natural-language extraction pipeline,
and real-time electricity grid carbon-intensity data to deliver personalized
carbon footprint estimates and explanations. Unlike static carbon calculators
that use fixed emission factors regardless of a user's actual grid or that
require structured form input, our system parses free-text lifestyle
descriptions and routes them through a hybrid predictor: an XGBoost model
(trained on 10,000 rows of a synthetically-generated benchmark dataset, R² =
0.83 on a held-out test set, R² = 0.8217 ± 0.0044 under 5-fold cross-validation)
serves as the primary predictor when an input falls within the model's trained
distribution, with a physically-grounded IPCC AR6 / Poore & Nemecek (2018)
emission-factor formula as the fallback otherwise. We report three empirical
findings from rigorously evaluating this system against realistic inputs
rather than only held-out rows from the training distribution. First, an
earlier production formula that used the physical formula as primary and
XGBoost as a bounded corrective adjustment scored R² = −73.8 (worse than
predicting the mean) — the inverse arrangement is required for this dataset.
Second, natural-language tool-call extraction reliability was measured at
progressively improving rates — 92% initial failure, reduced to a 43%
regression after a latency-driven pipeline rewrite, then 7.1%, then 0% in the
latest evaluation run — through three distinct, separately diagnosed fixes.
Third, and most significant: while XGBoost passes its numeric input-range
gate on 29.8% of realistic synthetic queries, accounting for the categorical
vocabulary the model actually saw during training (the raw dataset covers
only 3–10 of the 10 possible transport types, 4 of 13 food types, and 3 of 10
energy sources) drops the true safe usage rate to 0.8% — meaning the IPCC
formula, not XGBoost, is the de facto primary predictor for this deployed
system on realistic traffic. We report this and other limitations candidly, as
part of the paper's central contribution: an evaluation methodology for
hybrid ML–LLM systems that tests against realistic input distributions rather
than only in-distribution held-out data.

**Keywords** — carbon footprint estimation, hybrid ML pipeline, LLM tool
calling, XGBoost, out-of-distribution routing, real-time carbon intensity,
IPCC emission factors, evaluation methodology

## I. Introduction

Individual lifestyle choices — transportation, diet, and residential energy
use — contribute materially to global greenhouse gas emissions, and the IPCC
AR6 report identifies substantial per-capita emission reductions as necessary
to meet the 1.5°C target [1]. Existing carbon footprint calculators have three
recurring limitations: they use static, grid-agnostic emission factors; they
give generic advice regardless of whether a user's footprint is already low or
critically high; and they require structured form input rather than natural
language.

This paper describes a system that addresses these three limitations, and —
distinctly from most system papers in this space — reports what happened when
we evaluated it against realistic inputs rather than only inputs resembling
the training distribution. That evaluation surfaced two production bugs (one
a full accuracy inversion, one a silent categorical mis-encoding bug) and one
important, previously invisible fact about how rarely the trained ML model
actually engages on realistic traffic. We consider the evaluation methodology
and its results — not just the system design — to be this paper's primary
contribution, since it generalizes to any hybrid ML+rule-based system
deployed behind an LLM-driven natural-language interface.

Contributions:
- A hybrid carbon-prediction pipeline combining a trained XGBoost regressor,
  IPCC AR6 / Poore & Nemecek (2018) physical emission factors, and live grid
  carbon-intensity data, with an explicit, measured account of when each
  component is actually used in practice.
- A natural-language extraction pipeline using Groq-hosted
  `llama-3.1-8b-instant`, redesigned from an initial multi-tool ReAct loop
  into a fixed two-call pipeline for latency and API rate-limit reasons, with
  three iteratively diagnosed fixes to tool-call reliability documented as a
  case study in what actually causes `tool_use_failed` errors.
- An evaluation methodology — sampling realistic scenarios rather than
  held-out training rows — that finds a hybrid system's ML component may
  engage far less often than its held-out benchmark numbers suggest, and a
  demonstration of this on our own system (29.8% → 0.8% true engagement rate
  once categorical vocabulary coverage is accounted for).
- An honest accounting of the training dataset's synthetic provenance and its
  implications for interpreting the reported R².

## II. Related Work

**A. Carbon Footprint Calculators.** Static calculators such as the U.S. EPA's
Household Carbon Footprint Calculator [2] use fixed national-average emission
factors and do not adapt to a user's actual grid mix or personalize
suggestions to how far above or below average a user is. Wiedmann and Minx
[3] formalize the carbon-footprint concept and note that consumption-based
accounting is frequently omitted from simplified calculators.

**B. Emission Factor Sources.** We adopt IPCC AR6 Working Group III transport
emission factors [1] and the Poore and Nemecek (2018) food lifecycle emission
factors [4], derived from lifecycle analysis across 38,700 farms in 119
countries, directly in our formula-based fallback layer.

**C. Agentic AI and LLM Tool Use.** Tool-calling LLMs and agent orchestration
frameworks such as LangGraph [5] enable autonomous multi-step retrieval and
reasoning. Yao et al. [6] show ReAct-style agents outperform single-prompt
LLMs on tasks requiring iterative external tool use. We initially adopted this
paradigm and later replaced it with a fixed two-call pipeline once profiling
showed the autonomous multi-step loop was unnecessary overhead for a
workflow whose tool sequence is, in practice, deterministic once the primary
extraction is done (Section III.C) — a design tradeoff we report as a
finding, not just an implementation detail.

**D. Real-Time Carbon Intensity.** Electricity Maps [7] provides live carbon
intensity data for grid zones worldwide; using live rather than static
annual-average intensity is a natural refinement for any energy-emission
estimate, though we note in Section V that we can only characterize the
*magnitude* of this correction's effect on our held-out test set, not its
accuracy, since no ground-truth live-intensity value exists for historical
rows (Section V.B).

## III. System Architecture

### A. Data

The system's ML component is trained on the "Individual Carbon Footprint
Calculation" dataset (M. Duman et al., Kaggle, CC0) [8], 10,000 rows mapping
lifestyle attributes to an annual CO₂ figure. **This dataset is
synthetically generated**: its own listing states the target variable is
"calculated based on weightings from various studies and sites... attempting
to maintain values close to reality," not measured from real households. We
state this plainly because it changes what the reported R² means — see
Section V.A and Section VI. Food emission factors come from Poore and Nemecek
(2018) [4]; regional per-capita baselines come from Our World in Data,
covering 231 countries [9].

### B. Hybrid Prediction Pipeline

`predict_footprint` computes two independent estimates and selects between
them via a gate, rather than blending them:

1. **IPCC formula estimate.** `E_ipcc = km·EF_t·30 + kg_food·EF_f·30 +
   kWh·EF_g(I_live)·30 + flight_km/12·EF_flight + E_waste + E_clothing`,
   where `EF_g` uses live grid carbon intensity `I_live` (Electricity Maps,
   falling back to a static per-source baseline when the live API is
   unavailable).
2. **XGBoost estimate**, when the input passes an in-distribution gate
   (Section III.D): `E_final = E_ml + kWh·30·((I_live − I_base)/1000)`, i.e.
   the trained model's prediction, additively corrected for the delta between
   live and training-time grid intensity on the energy component only.

If the gate passes, `E_final` (path 2) is returned as the primary estimate;
otherwise `E_ipcc` (path 1, already live-grid-corrected) is returned. This is
the *opposite* arrangement from an earlier production version, which used the
IPCC formula as primary with XGBoost as a bounded ±15% adjustment — that
version scored R² = −73.8 against the held-out test set (Section V.A), because
the physical formula alone overestimates this dataset's target by roughly
2.5–3×. We report this inversion explicitly because it is easy to get backwards,
and got backwards once already in this project.

### C. LLM Orchestration Layer

Extraction and response generation use two sequential calls to
`llama-3.1-8b-instant` via the Groq API, not an autonomous multi-step agent
loop. Call 1 (temperature 0) extracts atomic lifestyle facts from the user's
message and invokes `predict_footprint`; call 2 (temperature 0.2) synthesizes
a natural-language response from the already-computed numeric results.
Three further tools — `get_live_carbon_intensity`, `compare_transport_scenarios`,
and `get_regional_baseline` — are invoked deterministically in code once the
prediction's inputs are known, rather than left to the LLM's own tool-choice
reasoning, since their arguments require no further judgment. This design was
adopted after an initial version using a four-tool ReAct loop (up to ~8 LLM
calls in the worst case, including a keyword-heuristic rerun) repeatedly
exhausted this project's Groq API tier (6,000 tokens/minute), producing
retry-backoff delays of 20–35+ seconds per request; the fixed pipeline both
removes that overhead and makes tool sequencing deterministic and auditable.

### D. In-Distribution Routing Gate

XGBoost is only used when the input is judged in-distribution, via two
checks: (1) each of five numeric features (`km_per_day`, `kg_food_per_day`,
`kwh_per_day`, `flights_per_year`, `flight_km_total`) must fall within the
range observed in the training data; (2) each of three categorical fields
(`transport_type`, `food_type`, `energy_source`) must be a category the
training data actually contained. Section V.C reports how often each check
passes on realistic input, and why check (2) — absent in an earlier version
of this system — mattered far more than check (1) alone suggested.

### E. Frontend and API

A FastAPI backend exposes `/chat` (conversational, session-aware, returns the
full tool-call trace), `/predict` (structured input), and `/health`. The
React frontend provides a chat interface with a live emissions breakdown
(donut chart via Recharts), a tool-call trace panel, and a footprint-threshold
badge. Session history is held in an in-memory store, a known scalability
limitation for multi-instance deployment.

### F. Threshold Mechanism

Suggestion tone is controlled by the *actual computed* footprint, not a fixed
script: below 200 kg CO₂/month triggers congratulatory framing with no
reduction suggestions; 200–375 kg/month triggers 1–2 gentle suggestions
targeting the largest emission category; 375–600 kg/month triggers full
actionable suggestions across categories; above 600 kg/month triggers urgent,
prioritized guidance. This directly conditions the LLM's synthesis prompt
rather than being inferred from the generated text, avoiding a
keyword-matching mismatch between stated tone and actual number.

## IV. Methodology

### A. Model Training

The XGBoost regressor (600 estimators, max depth 6, learning rate 0.04,
subsample 0.85, column subsample 0.85, min child weight 3, early stopping on
a held-out validation set) is trained on an 80/20 split (8,000/2,000 rows,
`random_state=42`) of the dataset described in Section III.A. Categorical
features are label-encoded; numeric features are standardized.

### B. Evaluation Suite

Three evaluation scripts, all included in this repository and re-runnable,
back every number reported in Section V:
- `evaluate_ablation.py` — accuracy of each pipeline configuration on the
  held-out test set, plus 5-fold cross-validation and a bootstrap confidence
  interval.
- `evaluate_agent.py` — 14 hand-authored natural-language test cases scored
  against the live Groq API for per-field extraction accuracy and tool-call
  reliability. Non-deterministic; re-run before citing.
- `evaluate_routing.py` — 3,000 randomly-sampled realistic scenarios (not
  training rows) scored against the routing gate, measuring how often the ML
  path is actually used and how much it disagrees with the formula fallback
  when both are computable.

## V. Results and Discussion

### A. Prediction Accuracy

| Model | MAE (kg CO₂/month) | R² |
|---|---|---|
| Mean baseline (sanity check) | 65.16 | 0.00 |
| IPCC formula only | 454.73 | −56.02 |
| RandomForest | 27.47 | 0.80 |
| **XGBoost (deployed primary)** | **25.30** | **0.83** (95% CI [0.81, 0.84]) |
| XGBoost, 5-fold CV (mean ± std) | 25.92 ± 0.24 | 0.8217 ± 0.0044 |

The tight cross-validation variance confirms the single-split R² was not a
lucky split. We highlight the IPCC-only row deliberately: an earlier
production version used the physical formula as primary with XGBoost as a
bounded ±15% adjustment and scored **R² = −73.8** on this same test set,
because the formula alone overestimates this dataset's target by roughly
2.5–3×. We surface this incident because getting the primary/fallback
direction backwards is an easy, costly mistake for any hybrid system in this
space, not specific to our implementation.

**Because the training target is synthetically generated** (Section III.A),
the R² above measures how well XGBoost recovers the dataset creators'
undisclosed weighting formula from the input features — not validated
real-world predictive accuracy. A model could be a near-perfect emulator of
that formula and still diverge from real emissions if the formula's own
weightings are imperfect approximations, which its own documentation hedges
("attempting to maintain values close to reality"). We consider this the
single most important caveat for interpreting every number in this table.

### B. Live Grid Fusion

Because the held-out test set has no associated "true" live grid intensity
at request time, we can only report the *magnitude* of the live-grid
correction term under an illustrative alternate-grid scenario, not its
accuracy. Under a scenario where grid intensity for India/US/EU drops from
their training-time baselines (708/386/276 gCO₂/kWh) to more decarbonized
current values (565/350/250), the mean correction is −0.87 kg CO₂/month
across the full test set and −3.48 kg CO₂/month among the 24.9% of rows using
a grid-tied energy source. We do not claim this improves accuracy — only
that it is a small, directionally sensible adjustment whose real-world
correctness is unverified.

### C. Routing Gate: How Often Is XGBoost Actually Used?

Over 3,000 randomly-sampled realistic scenarios (not training rows — see
`evaluate_routing.py`), 895/3,000 (29.8%) pass the five-feature numeric range
gate alone. `kwh_per_day` is the single largest numeric blocker (56.2% of
numeric-out-of-range cases), traced to a structural cause: this feature is
derived from device/shower hours in the raw dataset, not a full household
electricity bill, so its trained range is narrow by construction, not by
insufficient sampling. `flight_km_total` (33.9%) and `kg_food_per_day` (30.0%)
are comparably large contributors, traced to a different structural cause:
both are derived from small categorical lookup tables in the source data (4
flight-frequency buckets; 4 diet-type buckets), so a continuous, realistic
value for either was rarely something the model was trained to interpolate
over. We did not attempt to widen these ranges: a prior incident in this
project (documented internally) showed that widening `kwh_per_day`'s range
let a genuinely out-of-distribution input reach the model and produce a
materially wrong (understated) prediction.

**A second, larger gap:** the categorical fields had no vocabulary check at
all until this evaluation pass. The raw training data covers only 5 of 10
`transport_type` values, 4 of 13 `food_type` values, and 3 of 10
`energy_source` values that the extraction interface exposes — notably,
`grid_india`, the system's own default `energy_source`, was never a trained
category. Before this fix, an unrecognized category was silently re-mapped to
whichever class happens to be encoded as zero and *still returned an
XGBoost-labeled prediction* — a silent correctness bug, not merely an
accuracy gap. After adding this check, the true safe-usage rate falls to
**25/3,000 (0.8%)**: `energy_source` alone fails 70.5% of all sampled
scenarios, `food_type` 69.9%, `transport_type` 51.1%.

**Practical implication:** for this deployed system, on realistic
natural-language traffic, the IPCC formula — not the R²=0.83 XGBoost model —
is the de facto primary predictor. We consider this the paper's most
important empirical finding, ahead of the synthetic-data provenance issue,
because it means the model whose accuracy we report in Table above
essentially never runs on real queries. Among the 25 scenarios where XGBoost
does engage, it disagrees with the formula estimate by a mean of 581.53 kg
CO₂/month (69.3% relative; 84% of these differ by more than 50%) — not a
ground-truth comparison, since both methods estimate the same synthetic
target differently, but a measure of how much the routing decision can swing
the number a user sees.

Expanding the trained categorical vocabulary is a training-data coverage
limitation, not a routing-logic bug — the raw Kaggle dataset itself only
contains 3–5 category values per field to begin with — and is out of scope
without a richer, labeled dataset.

### D. Extraction Reliability

| Stage | Tool-call failure rate |
|---|---|
| Original design (LLM asked to pre-compute arithmetic) | 11/12 (92%) |
| After moving arithmetic into the tool | — (regression not yet reintroduced) |
| After ReAct→2-call pipeline rewrite (prose-listed categorical values) | 6/14 (43%) |
| After `Literal` type hints + `temperature=0` | 1/14 (7.1%) |
| After recovering rejected native-format tool calls | 0/14 (0%, latest run) |

Each row above is a *separately diagnosed and fixed* failure mode, not
repeated measurements of the same bug. Most notably: the 7.1% residual was
initially assumed to be inherent model flakiness, but direct repro showed one
specific query failed identically across four repeated calls at
`temperature=0` — deterministic, not random. Inspecting the raw Groq error
body showed the model's extraction was already fully correct; Groq had
rejected the response purely because it was emitted in Llama's native
`<function=name>{...}` text format rather than the expected structured
tool-call schema. Parsing and recovering that rejected text, rather than
retrying an identical request, resolved this case. General lesson: a
`tool_use_failed` rejection is a claim about response *format*, not
necessarily about extraction *correctness* — worth checking the raw error
body before assuming the model reasoned incorrectly.

## VI. Limitations

- **The training dataset's target is synthetically generated**, not measured
  real-world data (Section III.A, V.A) — the most important caveat for
  interpreting the reported R².
- **XGBoost's true safe engagement rate on realistic traffic is 0.8%**, not
  the 29.8% suggested by the numeric range gate alone (Section V.C); the
  IPCC formula is the practical primary predictor for this deployed system.
- **No external validation.** No comparison against an independent
  third-party carbon calculator or user study has been performed; an attempt
  to compare against a public calculator was abandoned after finding scope
  and unit mismatches too severe to produce trustworthy numbers.
- **The live-grid correction's accuracy is unverified** — only its magnitude
  under an illustrative scenario is reported (Section V.B).
- **The extraction LLM (`llama-3.1-8b-instant`) was chosen for cost/latency**,
  not benchmarked against larger models on extraction accuracy.
- **Reproducibility gap**: two of three trained model artifacts and the raw
  dataset are not committed to the repository; the deployed model artifact is
  committed and the application runs out of the box, but reproducing the
  training pipeline from a fresh clone requires separately sourcing the raw
  dataset.

## VII. Conclusion

We presented the AI Sustainability Copilot, a hybrid ML–LLM system for
personalized carbon footprint estimation, and — more centrally to this
paper's contribution — a rigorous, realistic-input evaluation of that system
that surfaced three significant, separately-fixed issues: a primary/fallback
inversion that had scored R² = −73.8, a series of natural-language tool-call
reliability regressions traced to distinct root causes, and a categorical
vocabulary gap that reduces the trained model's true safe usage rate from an
apparent 29.8% to 0.8% on realistic traffic. We report all three candidly,
including the last, because we believe under-testing a hybrid system's
routing behavior against realistic (rather than in-distribution) inputs is a
generalizable risk for this class of system, not one specific to ours.
Future work includes external validation against an independent carbon
calculator, expanding the training data's categorical coverage, and a
longitudinal user study of suggestion usefulness.

## References

[1] IPCC, "Climate Change 2022: Mitigation of Climate Change. Contribution of
Working Group III to the Sixth Assessment Report," Cambridge University
Press, 2022.

[2] U.S. EPA, "Household Carbon Footprint Calculator." [Online]. Available:
https://www.epa.gov/carbon-footprint-calculator

[3] T. Wiedmann and J. Minx, "A Definition of Carbon Footprint," in
*Ecological Economics Research Trends*, Nova Science Publishers, 2008, pp.
1–11.

[4] J. Poore and T. Nemecek, "Reducing food's environmental impacts through
producers and consumers," *Science*, vol. 360, no. 6392, pp. 987–992, Jun.
2018.

[5] LangChain, "LangGraph: Building Stateful, Multi-Actor Applications with
LLMs," 2024. [Online]. Available: https://langchain-ai.github.io/langgraph/

[6] S. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language
Models," in *Proc. ICLR 2023*, 2023.

[7] Electricity Maps, "Real-time Carbon Intensity API," 2024. [Online].
Available: https://www.electricitymaps.com/

[8] M. Duman et al., "Individual Carbon Footprint Calculation," Kaggle
Dataset, 2023, License CC0. [Online]. Available:
https://www.kaggle.com/datasets/dumanmesut/individual-carbon-footprint-calculation

[9] H. Ritchie, M. Roser, and P. Rosado, "CO₂ and Greenhouse Gas Emissions,"
Our World in Data, 2023.
