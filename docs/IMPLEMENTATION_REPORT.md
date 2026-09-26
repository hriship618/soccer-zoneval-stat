# PIVOT v1 implementation report

## Working stages connected to real data

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

The stage runner now passes canonical DFL events, tracking, and lineups into synchronized 1 Hz observations; generates event-only and tracking-augmented feature sets on identical samples; aggregates real player-match exposures; creates strictly historical profiles; constructs lineup impact rows when xG and profiles exist; evaluates held-out rows; and exports fitted ratings only when both fitting and evaluation succeed. Each blocked stage inspects its artifacts and reports the concrete missing input rather than returning an unconditional placeholder.

## Standalone helpers

- Match-disjoint chronological and expanding-window split utilities.
- Match-block bootstrap and match-aggregation utilities. These are implemented and tested as building blocks but are not reported as completed validation on the seven-match DFL sample.
- CUDA pitch-control acceleration remains a compute path for the legacy prototype, not a requirement or accuracy claim for PIVOT v1.

## Learned versus heuristic components

- Learned when adequate real inputs exist: shot occurrence, conditional future npxG, training-only context baselines, profile coefficients and player residuals.
- Heuristic: assumed ground-pass speed, player arrival speed, interception margin, pressure distance/relative-closing threshold, lane radius, squared attacking-x danger weight and transition window. Raw and danger-weighted lane coverage are stored separately. These remain configurable and require sensitivity analysis.
- Legacy only: the hand-written shot xG and defensive zone valuation in the original demo. They are never v1 training targets.

## Current data limitation

The local DFL/IDSSE release contains seven simultaneous final-matchday matches with tracking and events but no provider xG. Real feature extraction works, but the sample has no strictly earlier dates for profiles and no validated npxG targets. It is not sufficient for a credible chronological player-impact fit. The dashboard therefore shows no PIVOT player ratings.

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

## Validation status

The corrected World Cup event baseline has been retrained on a 38/13/13 chronological train/validation/test split. This validates only that event forecasting code outperforms or fails against its stated baselines on those partitions; it does not validate PIVOT player impact. PIVOT impact validation has not been performed because the DFL inputs lack xG and chronological history.

## Verification recorded on September 25, 2026

- `python -m pytest -q`: 22 tests passed, including executable artifact-stage fixtures.
- `python -m compileall -q zcpv scripts`: passed.
- `python -m scripts.train_zcpv features --match J03WMX`: processed 151,842 real tracking rows into 3,422 live observations, 32 player-match summaries, 346 turnover episodes, and 23 lineup segments.
- Real stage diagnostics: 222 shot-containing windows lack xG; 0 positive labeled windows remain; one processed match supplies no historical profile; 20 non-penalty shots lack xG for impact fitting.
- `python -m scripts.train_zcpv smoke`: completed a clearly labeled synthetic end-to-end fit and held-out evaluation; no fixture rating is exported to the dashboard.
- `python -m scripts.train_baselines`: regenerated the corrected event report using 38 training, 13 validation, and 13 test matches.
- Seven-match audit: 10,498 raw events and 6,371 parsed canonical events. The Köln–Bayern match reconciled the previously observed 951 passes, 21 shots, zero parsed carries and explicit reasons for every other exclusion.
- Köln–Bayern tracking audit at 1 Hz: 151,842 canonical rows, 145,942 player velocity rows and no player velocity above 14 m/s; period/gap boundaries do not produce velocities.
- `npx tsc --noEmit` and `npx oxlint app/dashboard.tsx`: passed.
- `npm run build`: passed. The repository-wide `npm run lint` still reports pre-existing violations in unused generated `components/ui/*` files and `hooks/use-mobile.ts`; none are in the changed dashboard file.

Generated raw/canonical data and large artifacts stay under ignored `data/raw/` and `data/processed/` paths. The small dashboard status export is the only committed generated v1 artifact.
