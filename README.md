# ZCPV Lab

Zonal Counterfactual Player Value is a research prototype for soccer player valuation from continuous tracking and event data. It combines transparent on-ball action value with leave-one-out pitch-control attribution, then exposes both components in an interactive dashboard.

## What is implemented

- 4 × 3 tactical zone mapping and xT-style fixed-point value iteration
- pass, carry, shot, take-on, interception, tackle, and regain valuation
- reaction- and velocity-adjusted pitch control
- exact leave-one-out attribution without materializing 22 extra control surfaces
- CUDA kernel that reduces directly into player × zone aggregates
- official DFL/IDSSE XML ingestion with synchronized event and tracking data
- responsive dashboard with match ranking, player detail, timeline, spatial heatmap, comparison, and methodology views

The dashboard shows the real 1. FC Köln 1–2 FC Bayern München match from May 27, 2023. Player names, minutes, event actions, and tracking samples come from the DFL/IDSSE open-data release (CC BY 4.0). ZCPV values are this project's research-prototype outputs, not official DFL ratings. The 12-zone action model is calibrated across all seven downloaded release matches; only the Bayern match is shown in the interface.

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

## Metric convention

For successful passes and carries, value is `V(destination) - V(origin)`. Failed actions lose the origin value. Shots use a small logistic xG model. Same-zone take-ons use avoided possession-loss risk. Defensive regains receive the opponent threat prevented. For every tracking frame, a player's spatial value is the zone-value-weighted loss in their team's pitch control when that player is removed. Match totals are normalized per 90 minutes.

## Data attribution

Match data: Deutsche Fußball Liga (DFL), licensed under CC BY 4.0. Dataset methodology: Bassek, Rein, Weber & Memmert (2025), DOI `10.1038/s41597-025-04505-y`. Raw release files are intentionally excluded from Git; the processed Bayern-match dashboard payload is committed for a reproducible hosted demo.
