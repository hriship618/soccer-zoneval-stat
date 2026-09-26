# PIVOT — Player Impact via Outcomes and Tracking

PIVOT is a real-data soccer player rating that combines a transferable event-value model with synchronized tracking counterfactuals. The production rankings use 64 FIFA World Cup 2022 matches from StatsBomb Open Data and seven DFL/IDSSE matches with continuous tracking. Synthetic data is restricted to tests and never enters the dashboard artifact.

## One-command pipeline

```bash
python -m pip install -e .
python -m scripts.run_pivot
```

That command:

1. chronologically splits the 64 World Cup matches into 38 train, 13 validation and 13 test matches;
2. fits scoring, conceding, shot-xG and xT models using only the training partition;
3. maps DFL actions into the same provider-neutral action representation;
4. aligns those actions to real DFL tracking frames and computes 32 × 21 pitch control plus every active player's leave-one-out counterfactual;
5. cross-fits the event/tracking fusion over seven leave-one-match-out DFL folds;
6. evaluates the held-out probabilities and simple player/team baselines;
7. writes real player-event contributions and final rankings under `data/processed/pivot/`;
8. regenerates `app/pivot-rankings.generated.ts`, the dashboard's only rating source.

Raw data is expected under `data/raw/statsbomb/` and `data/raw/dfl/`. Both raw and large processed artifacts are ignored by Git.

## Mathematical definition

For DFL event `e`, the frozen World Cup model supplies the actor-oriented net event change

```text
EV_e = [P_WC(score | post_e) - P_WC(concede | post_e)]
     - [P_WC(score | pre_e)  - P_WC(concede | pre_e)].
```

Two logistic fusion models are fitted on six DFL matches at a time from the World Cup probability logit and actor-oriented, xT-weighted pitch-control advantage. Removing player `i` from the synchronized frame changes that fused net outcome probability by `SCF_i,e`, oriented so a useful attacking or defensive presence is positive. The player-event contribution is

```text
C_i,e = 1[i performed e] * EV_e + SCF_i,e.
```

Displayed match contributions are always produced by the fold that held that match out. Player totals are divided by active synchronized event samples, expressed per 100 events, and shrunk toward the exposure-weighted population mean with reliability `N/(N+100)`. PIVOT is a 50/10 standardized rating; goalkeeper and outfield reference distributions are separate because their removal effects are structurally different. Players need at least 30 tracked minutes and 30 aligned samples.

## Event model

The target is whether the reference team scores or concedes within the next 10 actions. The current action is excluded, histories and labels never cross period boundaries, incomplete period-end windows are censored, and shootouts are removed. Features are the current action and two prior same-period actions: type, start/end coordinates, displacement, success, team relation, shot geometry, period, clock and score difference.

World Cup event files are never assigned tracking features. A separate World Cup shot-geometry model transfers StatsBomb xG to DFL shots for the xT comparison without fabricating provider xG.

## Tracking model

DFL tracking supplies position and velocity at the event time. PIVOT computes reaction-adjusted time-to-intercept control on a 32 × 21 grid, collapses it to 12 tactical zones, weights those zones with the World Cup-trained xT surface, and calculates the exact control change when each active player is removed. These are the only tracking-derived production features.

## Dashboard

```bash
npm install
npm run dev
```

The UI is intentionally a compact numerical ranking table: rank, player, team, minutes, synchronized sample count and PIVOT rating. It contains no placeholder cards, synthetic ratings or generated methodology prose.

## Validation and limitations

Run software checks with:

```bash
python -m pytest -q
python -m compileall -q zcpv scripts
npx tsc --noEmit
npm run build
```

Seven DFL matches are enough to demonstrate a leakage-safe, real-data end-to-end estimator, but not enough for a season-quality talent claim. World Cup-to-Bundesliga transfer can have competition/provider shift; pitch-control physics are assumptions; DFL outcomes are sparse; and the standardized rating is local to this player pool. The generated `pivot-report.json` records exact held-out metrics and does not claim tracking improves every metric when it does not.

## Data attribution

Tracking/event data: Deutsche Fußball Liga, DFL/IDSSE open data, CC BY 4.0. Dataset methodology: Bassek, Rein, Weber & Memmert (2025), DOI `10.1038/s41597-025-04505-y`.

World Cup events and xG labels: StatsBomb Open Data. Published uses must identify StatsBomb as the source and follow the attribution requirements in the [official repository](https://github.com/hudl/open-data).
