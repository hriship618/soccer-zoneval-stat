# ZCPV Lab

Zonal Counterfactual Player Value is a research prototype for soccer player valuation from continuous tracking and event data. It combines transparent on-ball action value with leave-one-out pitch-control attribution, then exposes both components in an interactive dashboard.

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

The dashboard shows the real 1. FC Köln 1–2 FC Bayern München match from May 27, 2023. Player names, minutes, event actions, and tracking samples come from the DFL/IDSSE open-data release (CC BY 4.0). ZCPV values are this project's research-prototype outputs, not official DFL ratings. Zone values come from the trained StatsBomb World Cup 2022 xT surface, area-averaged from 16 × 12 to the dashboard's 4 × 3 tactical grid. No hand-shaped shot or goal counts are injected into the fit.

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

This version is a single-match tracking study, not a scouting grade. The trained World Cup xT surface now drives the displayed zone values, but shots and defensive actions remain partly heuristic, per-90 values are unstable for short appearances, and uncertainty intervals are not yet implemented. The dashboard exposes these limitations instead of treating the ranking as ground truth.

## Data attribution

Match data: Deutsche Fußball Liga (DFL), licensed under CC BY 4.0. Dataset methodology: Bassek, Rein, Weber & Memmert (2025), DOI `10.1038/s41597-025-04505-y`. Raw release files are intentionally excluded from Git; the processed Bayern-match dashboard payload is committed for a reproducible hosted demo.

Event-model data: StatsBomb Open Data. Published uses must identify StatsBomb as the data source and follow the attribution requirements in the [official repository](https://github.com/hudl/open-data).
