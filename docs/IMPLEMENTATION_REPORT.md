# PIVOT implementation report

## Production path

`python -m scripts.run_pivot` trains the World Cup event models, processes all seven DFL tracking matches, cross-fits DFL fusion, evaluates held-out predictions, writes player-event/ranking artifacts, and regenerates the dashboard input. The dashboard imports only `app/pivot-rankings.generated.ts` and shows separate outfield and goalkeeper rankings.

## Current computed data

- 64 World Cup matches: 38 train, 13 validation, 13 test.
- Seven DFL matches: 6,004 common actions, 5,604 live tracking alignments, and 5,480 complete next-10-action evaluation horizons.
- Per-match alignment rates range from 88% to 98%.
- 120 players qualify with at least 45 tracked minutes.
- Every production player-event contribution is leave-one-match-out cross-fitted.
- Ratings are seven-match sample ratings, not estimates of season-long ability.

## Tracking ablation

All values below concatenate the seven held-out folds. Event-only is a DFL calibration of the frozen World Cup event probability; event plus control adds only actor-oriented pitch-control advantage.

| Target | Model | Brier | Log loss | ROC AUC |
| --- | --- | ---: | ---: | ---: |
| Score | Constant | 0.021806 | 0.107584 | 0.3673 |
| Score | Event only | **0.021703** | **0.105344** | **0.5993** |
| Score | Event + control | 0.021716 | 0.105610 | 0.5941 |
| Concede | Constant | 0.008875 | 0.051847 | 0.3128 |
| Concede | Event only | **0.008858** | **0.050780** | 0.5675 |
| Concede | Event + control | 0.008872 | 0.051076 | **0.5749** |

Adding control does not improve score prediction on any reported metric. For conceding it improves AUC but slightly worsens Brier and log loss. Therefore this sample does not support a general claim that tracking improves predictive performance.

## Fold-level control coefficients

These are coefficients in original pitch-control-advantage units; the generated report also stores standardized coefficients.

| Held-out match | Score coefficient | Concede coefficient |
| --- | ---: | ---: |
| J03WMX | +4.70 | -7.11 |
| J03WN1 | +2.82 | -3.01 |
| J03WOH | +4.80 | -9.79 |
| J03WOY | +2.50 | -4.63 |
| J03WPY | +2.68 | -5.19 |
| J03WQQ | +2.64 | -4.72 |
| J03WR9 | +1.42 | -3.73 |

Signs are stable in all seven folds: more actor-oriented control raises scoring probability and lowers conceding probability. Magnitudes are only moderately stable: score mean `+3.08` (SD `1.14`, range `+1.42` to `+4.80`) and concede mean `-5.45` (SD `2.13`, range `-9.79` to `-3.01`). This supports directional spatial attribution, not a strong predictive-improvement claim.

## Conservative exposure adjustment

Raw PIVOT per 100 synchronized events remains in every ranking row. Rating reliability is now:

```text
effective matches = min(matches, minutes / 90)
reliability       = effective matches / (effective matches + 4)
```

A full single match receives reliability `0.20`, even with 900 synchronized frames. Five full matches receive `0.556`. The rating uses the unshrunk raw-rate standard deviation as its fixed scale; otherwise standardizing after shrinkage would expand the compressed distribution and negate the adjustment.

## Current outfield top five

1. D. Ginczek — 55.32
2. Serge Gnabry — 55.04
3. Kingsley Coman — 54.66
4. Leroy Sané — 54.62
5. M. Diaby — 54.43

## Current goalkeeper top five

1. D. Stojanovic — 53.62
2. M. Schwäbe — 53.36
3. Y. Sommer — 52.11
4. C. Mathenia — 50.97
5. N. Vasilj — 49.71

All ranking inputs are real. Synthetic fixtures are restricted to tests. Detailed ignored artifacts live under `data/processed/pivot/`; the generated TypeScript rankings are committed for Vercel.
