# PIVOT real-data methodology

The production estimator is implemented by `python -m scripts.run_pivot`. It keeps event-only World Cup data and DFL tracking data separate until inference in their genuinely shared action space.

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

Aggregate all held-out player-event contributions, divide by active synchronized event samples and express the rate per 100 samples. Empirical-Bayes shrinkage uses reliability `N_i/(N_i+100)`. The display rating is `50 + 10z`, standardized separately for goalkeepers and outfield players because their removal distributions are structurally different. Qualification requires 30 tracked minutes and 30 event samples.

## Evaluation

The World Cup test partition reports Brier score, log loss, ROC AUC and calibration error. DFL evaluation concatenates predictions from the seven held-out folds and compares constant, transferred World Cup and tracking-fusion probabilities. A secondary 14 team-match diagnostic compares raw event totals, minutes-adjusted event value, transferred xT and PIVOT against goal difference; it is explicitly descriptive because the sample is very small.

## Scope

These are real, reproducible rankings for the supplied matches, not season-long talent estimates or causal effects. Important limitations are event-model domain shift, sparse DFL goal labels, pitch-control assumptions, dependence among matches, and a rating scale local to this player pool. Exact results are written to `data/processed/pivot/pivot-report.json` on every run.
