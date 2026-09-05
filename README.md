# ZCPV Lab

Zonal Counterfactual Player Value is a research prototype for soccer player valuation from continuous tracking and event data. It combines transparent on-ball action value with leave-one-out pitch-control attribution, then exposes both components in an interactive dashboard.

## What is implemented

- 4 × 3 tactical zone mapping and xT-style fixed-point value iteration
- pass, carry, shot, take-on, interception, tackle, and regain valuation
- reaction- and velocity-adjusted pitch control
- exact leave-one-out attribution without materializing 22 extra control surfaces
- CUDA kernel that reduces directly into player × zone aggregates
- official DFL/IDSSE XML ingestion with synchronized event and tracking data
- deterministic synthetic tracking generator, honest CPU/GPU benchmark harness, and tests
- responsive dashboard with match ranking, player detail, timeline, spatial heatmap, comparison, and methodology views

The dashboard shows the real 1. FC Köln 1–2 FC Bayern München match from May 27, 2023. Player names, minutes, event actions, and tracking samples come from the DFL/IDSSE open-data release (CC BY 4.0). ZCPV values are this project's research-prototype outputs, not official DFL ratings. The 12-zone action model is calibrated across all seven downloaded release matches; only the Bayern match is shown in the interface.

## Run the dashboard

```bash
npm install
npm run dev
```

## Run the analytics engine

```bash
python -m pip install -e .[test]
python -m pytest
python scripts/crunch_dfl.py --match J03WMX
python scripts/generate_demo.py
python scripts/benchmark.py --frames 10 50 100
```

CUDA is optional because development machines may not have an NVIDIA GPU. On CUDA 12, install `.[cuda]`; the benchmark automatically measures the GPU path when available and records `null` otherwise. Results written by `scripts/benchmark.py` are measurements from the machine that ran it, never estimates.

## Metric convention

For successful passes and carries, value is `V(destination) - V(origin)`. Failed actions lose the origin value. Shots use a small logistic xG model. Same-zone take-ons use avoided possession-loss risk. Defensive regains receive the opponent threat prevented. For every tracking frame, a player's spatial value is the zone-value-weighted loss in their team's pitch control when that player is removed. Match totals are normalized per 90 minutes.

## Scope

This is an internship/research portfolio prototype, not a validated production scouting metric. Its novelty is the unified attribution layer and visible decomposition. Credible next steps are stability tests across sampling rates and grid resolutions, ablations against action-only models, and out-of-sample prediction of possession and shot outcomes.

## Data attribution

Match data: Deutsche Fußball Liga (DFL), licensed under CC BY 4.0. Dataset methodology: Bassek, Rein, Weber & Memmert (2025), DOI `10.1038/s41597-025-04505-y`. Raw release files are intentionally excluded from Git; the processed Bayern-match dashboard payload is committed for a reproducible hosted demo.
