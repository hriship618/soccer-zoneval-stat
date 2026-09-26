# PIVOT v1 research methodology

PIVOT (Player Impact via Outcomes and Tracking) v1 is a proposed tracking-informed soccer impact metric. It combines on-ball actions with spatial measurements of receiving availability, pressure, lane coverage and transition protection. Models trained on held-out match sequences estimate each side's near-term non-penalty expected goals. Lagged player measurements then inform a regularized lineup model that estimates O-PIVOT, D-PIVOT, and net non-penalty xG impact per 90.

This is a research prototype. Its first question is whether tracking information improves out-of-time prediction beyond event-only baselines. It does not claim comprehensive, replacement-level or causal player value.

## Evidence chain

1. The DFL adapter normalizes provider metadata, events, sampled tracking and lineup intervals. Physical and attacking coordinates are both retained; provider directions are preferred and inferred fallbacks are recorded.
2. Live-play states are sampled at 1 Hz. The team in possession becomes the fixed reference team for the full future window, including after turnovers.
3. Targets sum provider non-penalty xG in `(t, t + 15s]` separately for the reference team and opponent. Windows do not cross periods; unknown possession, ambiguous shot synchronization and right-censored states are excluded.
4. Event-context and tracking-augmented situation models use identical match-level partitions. A shot-occurrence classifier times a nonnegative conditional xG regressor is compared with direct regression.
5. Player-match measurements retain exposure, volume, effectiveness, units and missingness. Profiles use earlier matches only, exponential recency weighting and role-aware shrinkage based on episode or match exposure—not frame count.
6. Substitution, dismissal and period boundaries define lineup segments. Two attacking-perspective rows from one segment remain in the same fold.
7. The impact model predicts segment npxG rate from context, summed lagged profiles and regularized player residuals. Positive defense means less npxG conceded. Offense plus defense equals net.

Physical DFL coordinates use metres from the home-left corner (`x=0..105`, `y=0..68`). Feature extraction rotates every entity and velocity into the reference team's attacking-right coordinate system. Open-play receiver eligibility applies the ball, halfway line, and second-last-opponent offside tests; goal kicks, throw-ins, and corners are exempt. Lane danger uses the squared attacking-x fraction as an explicitly heuristic weight and is stored separately from raw coverage.

The StatsBomb event baseline is a post-action forecast: features may include the completed current action, but its outcome is never a future label. The label window begins with the next action, remains inside the same period, and is censored unless ten later same-period actions exist. Period-five shootout events are excluded.

## Interpretation boundaries

- Receiver feasibility is a configurable ground-pass arrival/interception heuristic, not a calibrated completion probability.
- Pressure, lane coverage and transition protection are experimental proxies, not independently verified causal contributions.
- Team-season sensitivity controls compete with player effects and cannot guarantee causal separation.
- Goalkeepers remain lineup controls but are excluded from public outfield rankings. Their coefficient is not shot-stopping ability.
- Negative predicted rates remain visible in evaluation rather than being clipped to improve reported metrics.

## Availability rule

Production fitting is blocked unless the data provide non-penalty xG or a documented frozen xG model, enough positive windows, chronological partitions, valid lineups and adequate lineup variation. Synthetic fixtures verify code behavior only. Null estimates are exported with a reason; they are never replaced by zero.
