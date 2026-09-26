# PIVOT implementation report

## Production path

`python -m scripts.run_pivot` is the complete real-data entry point. It trains the World Cup event models, processes all seven DFL tracking matches, cross-fits the DFL fusion, evaluates held-out predictions, writes player-event and ranking artifacts, and regenerates the dashboard input. The dashboard imports only `app/pivot-rankings.generated.ts`; legacy and insufficient-data exports are not part of the UI.

## Current computed data

- 64 World Cup matches: 38 train, 13 validation, 13 test.
- Seven DFL matches: 6,004 common actions, 5,604 live tracking alignments, and 5,480 complete next-10-action evaluation horizons.
- Per-match alignment rates range from 88% to 98%.
- 130 players qualify with at least 30 tracked minutes and 30 synchronized samples.
- Every production player-event contribution is leave-one-match-out cross-fitted.

## Held-out results

On World Cup test data, scoring has Brier `0.00982` and ROC AUC `0.7552`; conceding has Brier `0.00222` and ROC AUC `0.7846`.

On concatenated held-out DFL folds, tracking fusion has scoring Brier `0.02172`, log loss `0.10561`, and ROC AUC `0.5941`, versus the raw transferred event model's `0.02199`, `0.12627`, and `0.6298`. Conceding fusion has Brier `0.00887`, log loss `0.05108`, and ROC AUC `0.5749`, versus `0.00891`, `0.06022`, and `0.6180`. Thus fusion improves calibration/probabilistic loss here but reduces discrimination; the implementation does not claim universal superiority.

The 14-observation team-match comparison is retained only as a small-sample diagnostic. The generated report contains its exact baseline correlations and warning.

## Current top five

1. Kingsley Coman — 76.84
2. D. Ginczek — 76.35
3. Serge Gnabry — 74.75
4. Leroy Sané — 72.65
5. M. Diaby — 70.90

All names, teams, minutes, events, tracking samples and ranking inputs are derived from the supplied real datasets. Synthetic fixtures are used only by unit/smoke tests.

## Reproducibility and checks

The real pipeline writes detailed ignored artifacts under `data/processed/pivot/`, including models, player-event contributions, evaluation, rankings and the complete report. The small generated TypeScript ranking payload is committed for Vercel.

Software verification covers 24 Python tests, Python bytecode compilation, TypeScript type checking, focused application lint and a production web build.
