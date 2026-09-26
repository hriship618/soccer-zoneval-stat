# PIVOT real-data methodology

The production estimator is implemented by `python -m models.scripts.run_pivot`. It keeps event-only World Cup data and DFL tracking data separate until inference in their genuinely shared action space.

## World Cup event model

The 64 FIFA World Cup 2022 StatsBomb Open Data matches are sorted chronologically and split by match: 38 train, 13 validation and 13 test. Penalty shootouts are excluded. A state is observed after an action; its binary targets are a goal scored and a goal conceded in the next 10 same-period actions. The current action is excluded and incomplete period-end horizons are censored.

The current action and two prior same-period actions supply type, normalized start/end coordinates, displacement, completion, relation to the current team and shot geometry. Clock, period and score difference are state context. Histogram gradient boosting estimates `P(score)` and `P(concede)`. A separate geometry regressor transfers StatsBomb shot xG to DFL shots for the xT baseline. No World Cup record receives tracking features.

For DFL event `e`, actor-oriented on-ball value is

```text
EV_e = (P_WC(score|post_e) - P_WC(concede|post_e))
     - (P_WC(score|pre_e)  - P_WC(concede|pre_e)).
```

When possession changes, the prior net state is sign-flipped into the new actor's frame.

## DFL tracking model

Provider directions rotate each DFL action so the actor attacks left-to-right, then metres are scaled into the World Cup 120 × 80 action representation. Each event is joined only to a real, live DFL tracking frame from the same period.

At that frame, reaction- and velocity-adjusted time to intercept creates pitch control on a 32 × 21 grid. Control is aggregated into 12 tactical zones and weighted by the World Cup-trained xT surface in the actor's attacking direction. Exact algebraic leave-one-player-out recomputation provides each active player's counterfactual control loss.

## Cross-fitted fusion and attribution

Seven leave-one-match-out folds fit two regularized logistic calibrators on six DFL matches and apply them to the seventh. Inputs are the World Cup score/concede probability logit and actor-oriented threat-weighted team control advantage. The label is again scoring or conceding in the next 10 same-period DFL actions. Consequently, no displayed contribution comes from a model fitted on its own match.

For player `i`, remove their pitch-control mass and recompute the fused net probability. Orient the change to the player's own team to obtain `SCF_i,e`. Then

```text
C_i,e = 1[i is the actor] EV_e + SCF_i,e.
```

Aggregate all held-out player-event contributions, divide by active synchronized event samples and retain that raw rate per 100 samples. Frames within one match are strongly correlated and therefore do not determine uncertainty. Instead:

```text
effective_matches_i = min(matches_i, minutes_i / 90)
reliability_i       = effective_matches_i / (effective_matches_i + 4)
shrunk_rate_i       = reliability_i * raw_rate_i
                    + (1 - reliability_i) * group_prior
```

The four-match prior is deliberately conservative for a seven-match dataset. A full one-match player has reliability `0.20`, a 45-minute player has `0.111`, and five full matches have `0.556`, regardless of synchronized frame count. The display rating is `50 + 10 * (shrunk_rate - group_prior) / SD(raw group rates)`. Using the unshrunk standard deviation is essential: re-standardizing on the compressed post-shrink distribution would undo the uncertainty adjustment. Goalkeepers and outfield players have separate priors, scales and ranking tables. Qualification requires at least 45 tracked minutes.

## Evaluation

The World Cup test partition reports Brier score, log loss, ROC AUC and calibration error. DFL evaluation concatenates predictions from the seven held-out folds. Each fold fits a constant baseline, an event-only DFL calibration of the frozen World Cup probability, and the same calibration plus pitch-control advantage. This is a direct tracking ablation: every alternative uses identical held-out matches and labels. Both standardized and original-unit control coefficients are recorded per fold, along with sign-stability summaries. A secondary 14 team-match diagnostic is explicitly descriptive because the sample is very small.

## Scope

These are real, reproducible **seven-match sample ratings**, not estimates of season-long player ability or causal effects. Important limitations are event-model domain shift, sparse DFL goal labels, pitch-control assumptions, dependence among matches, and a rating scale local to this player pool. Exact results are written to `models/data/processed/pivot/pivot-report.json` on every run.
