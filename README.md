# ZCPV Lab

Zonal Counterfactual Player Value is a research prototype for tracking-informed soccer impact analysis. The repository now keeps two explicitly separate paths:

- `legacy_v0`: the original single-match action plus pitch-control demo, retained unchanged in purpose.
- `zcpv-v1-research`: an auditable pipeline toward offensive and defensive non-penalty xG impact per 90. It refuses to publish ratings when target or sample requirements are not met.

## What is implemented

- 4 × 3 tactical zone mapping and xT-style fixed-point value iteration
- pass, carry, shot, take-on, interception, tackle, and regain valuation
- reaction- and velocity-adjusted pitch control
- exact leave-one-out attribution without materializing 22 extra control surfaces
- CUDA kernel that reduces directly into player × zone aggregates
- official DFL/IDSSE XML ingestion with synchronized event and tracking data
- StatsBomb Open Data ingestion for the 64-match FIFA World Cup 2022 release
- trained 16 × 12 xT and VAEP-style scoring/conceding probability baselines
- chronological match-level validation with Brier score, log loss, ROC AUC, and calibration error
- responsive dashboard with match ranking, player detail, timeline, spatial heatmap, comparison, and methodology views
- canonical DFL matches, events, tracking, lineups and state-sample schemas
- raw-to-parsed event audits, exclusion reasons, checksums and match quality reports
- leakage-safe 15-second target construction with fixed reference-team identity
- heuristic receiving feasibility, pressure, lane-coverage and transition-protection features
- lagged exposure-aware player profiles and role-aware shrinkage
- duration-weighted offensive/defensive impact estimator with separate regularization
- match-level splits, evaluation helpers and match-block bootstrap infrastructure

The dashboard shows the real 1. FC Köln 1–2 FC Bayern München match from May 27, 2023. Player names, minutes, event actions, and tracking samples come from the DFL/IDSSE open-data release (CC BY 4.0). The visible match ranking is labeled `legacy_v0`; it is descriptive and is not the v1 player-impact model. The v1 research tab exposes its current `insufficient_data` status instead of fabricated zero ratings.

## Run the dashboard

```bash
npm install
npm run dev
```

## Run the analytics engine

```bash
python -m pip install -e .
python scripts/crunch_dfl.py --match J03WMX
```

## Run the v1 research pipeline

All generated canonical tables, quality reports and model artifacts are written under ignored `data/processed/` paths. JSONL is the dependency-free canonical interchange; `schema.json` records exact fields and coordinate/time conventions.

```bash
python -m scripts.train_zcpv audit
python -m scripts.train_zcpv features
python -m scripts.train_zcpv state
python -m scripts.train_zcpv profiles
python -m scripts.train_zcpv impact
python -m scripts.train_zcpv evaluate
python -m scripts.train_zcpv export
python -m scripts.train_zcpv smoke
```

Run every locally executable stage with:

```bash
python -m scripts.train_zcpv all
```

Add `--include-tracking` to the audit command to materialize sampled 1 Hz tracking JSONL. This is deliberately opt-in because it is large. The smoke stage uses synthetic fixtures only to prove software invariants; its outputs are never shown as real estimates.

## Train the event-data baselines

The fetcher downloads the official StatsBomb World Cup 2022 event release. Raw files and fitted binary models stay outside Git; the small validation report used by the dashboard is committed as generated TypeScript.

```bash
python scripts/fetch_statsbomb.py
python -m scripts.train_baselines
```

The current split trains on 51 chronologically earlier matches and evaluates on 13 later matches beginning December 4, 2022. The held-out matches never fit the xT grid or either VAEP-style probability model. The two fitted gradient-boosting models and xT grid are written to `data/processed/models/`. Running `crunch_dfl.py` afterward automatically uses that trained xT grid for the displayed DFL score; if it is absent, the script explicitly falls back to the seven-match DFL fit.

## Recorded performance

Recorded on the real Köln–Bayern match (17,071 live tracking frames at 5 Hz; 32 × 21 pitch-control grid) using an AMD Ryzen 9 5900X 12-Core Processor and an NVIDIA GeForce RTX 3080.

| Path | Time | Speedup |
| --- | ---: | ---: |
| CPU naive | 47.32 s | 1.0× |
| CPU optimized | 6.45 s | 7.3× |
| CUDA (RTX 3080) | 47.5 ms | 135.7× vs. optimized CPU; ~1000× vs. naive |

## Metric convention and limitations

For successful passes and carries, value is `V(destination) - V(origin)`. Failed actions lose the origin value. Shots use a small logistic xG model. Same-zone take-ons use avoided possession-loss risk. Defensive regains receive the opponent threat prevented. For every tracking frame, a player's spatial value is the zone-value-weighted loss in their team's pitch control when that player is removed. Match totals are normalized per 90 minutes.

Legacy v0 is a single-match tracking study, not a scouting grade. The trained World Cup xT surface drives its displayed zone values, but shots and defensive actions remain partly heuristic and per-90 values are unstable for short appearances.

V1 requires provider xG or a separately trained and frozen calibrated xG model plus substantially more chronological lineup variation. The seven DFL matches support ingestion checks and exploratory tracking features, not credible season rankings. When those requirements are absent, fitting stages return `insufficient_data`. See [the v1 methodology](docs/METHODOLOGY_V1.md) and [implementation report](docs/IMPLEMENTATION_REPORT.md).

## Data attribution

Match data: Deutsche Fußball Liga (DFL), licensed under CC BY 4.0. Dataset methodology: Bassek, Rein, Weber & Memmert (2025), DOI `10.1038/s41597-025-04505-y`. Raw release files are intentionally excluded from Git; the processed Bayern-match dashboard payload is committed for a reproducible hosted demo.

Event-model data: StatsBomb Open Data. Published uses must identify StatsBomb as the data source and follow the attribution requirements in the [official repository](https://github.com/hudl/open-data).
