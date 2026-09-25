# ZCPV v1 implementation report

## Implemented

- Versioned configuration, canonical dataclasses and schema manifest.
- DFL adapter with explicit XML-node selection, raw/parsed counts, exclusion reasons, stable identifiers, provider direction provenance, source timestamps and sampled tracking.
- Actual-time velocity calculation that never crosses periods or large gaps; missing positions and velocities stay missing.
- Lineup intervals, period-aware segments and explicit 11-v-11 eligibility.
- Fixed-reference 15-second target construction with penalty, synchronization and censoring rules.
- Present-time spatial state features plus heuristic receiver, pressure, lane-coverage and transition-protection measurements.
- Two-stage state-value estimator and direct-regression baseline with availability checks and appropriate occurrence/xG metrics.
- Exposure-aware lagged profiles, role shrinkage, recency weighting and low-information flags.
- Duration-weighted offensive/defensive impact estimator with separate coefficient/player penalties, training-only standardization, unseen-player handling and exposure-weighted centering.
- Chronological/expanding match splits, match aggregation, regression metrics and refitted match-block bootstrap utility.
- Honest versioned export and dashboard research view. The old match demo is visibly labeled `legacy_v0`.

## Learned versus heuristic components

- Learned when adequate real inputs exist: shot occurrence, conditional future npxG, context baselines, profile coefficients and player residuals.
- Heuristic: assumed ground-pass speed, player arrival speed, interception margin, pressure distance/closing threshold, lane radius and transition window. These remain configurable and must receive sensitivity analysis.
- Legacy only: the hand-written shot xG and defensive zone valuation in the original demo. They are never v1 training targets.

## Current data limitation

The local DFL/IDSSE release contains seven complete matches with tracking and events but no provider xG. It is suitable for ingestion audits and leave-one-match-out exploratory diagnostics. It is not sufficient for a credible chronological player-impact fit. The v1 state, profile, impact and evaluation stages therefore report `insufficient_data`; the dashboard shows no v1 player ratings.

The exact remaining inputs for meaningful training are:

1. A larger chronological set of synchronized events, tracking and authoritative lineups with stable player/team IDs.
2. Provider non-penalty shot xG, or a separately trained, calibrated and frozen shot-xG model with documented source and cutoff.
3. Enough positive 15-second windows and lineup variation for match-level train/validation/test partitions.
4. Scores, numerical advantage, dismissals, period boundaries and authoritative attacking directions at observation time.

## Commands

```bash
python -m pip install -e .
python -m scripts.train_zcpv audit
python -m scripts.train_zcpv features
python -m scripts.train_zcpv smoke
python -m scripts.train_zcpv export
python -m pytest
npm run lint
npm run build
```

## Verification recorded on September 25, 2026

- `python -m pytest -q`: 12 tests passed.
- `python -m compileall -q zcpv scripts`: passed.
- `python -m scripts.train_zcpv all`: all audit/infrastructure stages completed; scientific fit stages returned the expected `insufficient_data` status.
- Seven-match audit: 10,498 raw events and 6,371 parsed canonical events. The Köln–Bayern match reconciled the previously observed 951 passes, 21 shots, zero parsed carries and explicit reasons for every other exclusion.
- Köln–Bayern tracking audit at 1 Hz: 151,842 canonical rows, 145,942 player velocity rows and no player velocity above 14 m/s; period/gap boundaries do not produce velocities.
- `npx tsc --noEmit` and `npx oxlint app/dashboard.tsx`: passed.
- `npm run build`: passed. The repository-wide `npm run lint` still reports pre-existing violations in unused generated `components/ui/*` files and `hooks/use-mobile.ts`; none are in the changed dashboard file.

Generated raw/canonical data and large artifacts stay under ignored `data/raw/` and `data/processed/` paths. The small dashboard status export is the only committed generated v1 artifact.
