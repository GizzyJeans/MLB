# mlbline

Quantitative analysis of MLB full-game **run line** and **total** markets.

The library does three separable things, and keeping them separate is the
whole design:

| Module | Answers |
|---|---|
| `devig` | What probability is a bookmaker's price actually implying, once margin is removed? |
| `market` | Is any book out of step with the rest of the market? |
| `rundist` | What does an independent estimate of the game say the probability is? |
| `policy` | Is the resulting edge large enough, and complete enough, to bet? |

`market` and `rundist` produce numbers that look identical on a screen and
mean completely different things. Beating the consensus is a line-shopping
edge: real, small, and it evaporates as books correct. Beating the market
requires a view on the game itself. A report that blurs the two will
eventually stake real money on the first while claiming the authority of the
second, so nothing in this codebase lets a market-derived probability stand
in for a model one — `policy.evaluate` rejects the substitution explicitly.

## Current status: the model cannot be run

`rundist` is implemented, calibrated and tested, but it needs per-side
expected runs, and those need pitcher, lineup, bullpen, park and weather
inputs. **In this environment every source of those inputs is unreachable.**
The egress policy permits the odds feed and GitHub; it denies everything else:

```
statsapi.mlb.com          blocked      baseballsavant.mlb.com    blocked
www.fangraphs.com         blocked      www.baseball-reference.com blocked
www.rotowire.com          blocked      www.mlb.com / espn.com    blocked
api.weather.gov           blocked      api.open-meteo.com        blocked
api.the-odds-api.com      OK (key in ODDS_API_KEY)
```

So the pipeline runs end to end on market data and stops at the gate, which
is the correct outcome rather than a degraded one. See
`reports/2026-08-08_market_scan.md`.

To make it operational, supply expected runs per side from any source and the
rest of the chain works unchanged:

```python
from mlbline.rundist import simulate_game

sim = simulate_game(expected_home_runs=4.9, expected_away_runs=4.1)
sim.prob_cover(-1.5, home=True)   # home run line
sim.prob_over(8.5)                # total
```

## Why the simulation is half-inning based

Three rules move run line and total prices by more than most handicapping
edges are worth, and a model that draws two game totals from a pair of
Poissons gets all three wrong:

1. **The home team does not bat in the bottom of the 9th when it leads.**
   This truncates precisely the winning margins a `-1.5` line is priced on.
   At equal strength the engine has a home team covering `-1.5` 31.2% of the
   time, matching the observed 30–32%; a symmetric model says ~35% and will
   systematically overpay for home favourites.
2. **A walk-off ends play on the go-ahead run**, capping the margin at one
   unless the hit clears the fence.
3. **Extra innings start with a runner on second**, fattening the over tail.

Run scoring per half-inning is a Gamma-Poisson mixture, not a Poisson. Real
team scores have a variance-to-mean ratio near 2.1 because big innings
cluster; pure Poisson forces 1.0 and badly understates both tails.

## Usage

```bash
python3 scripts/scan_market.py --fetch --out reports/today.md --all-lines
python3 scripts/scan_market.py --snapshot data/snapshots/<file>.json
python3 tests/test_model.py
```

`--all-lines` also prices the mirror run line and alternate totals. Note that
books split on which side they hang the run line from: some quote
`Home -1.5 / Away +1.5`, others `Home +1.5 / Away -1.5`. Those are two
different markets — a ~35% outcome and a ~65% one — and pooling them by
absolute value manufactures spectacular phantom edges. `Quote.line_key`
references the point to the home team to keep them apart.

## Staking policy

Encoded in `policy.py` so a recommendation can be audited:

- minimum **+4%** expected value on the model's own numbers
- minimum **3 percentage points** of divergence from the de-vigged market
- **0.25 Kelly**, capped at **1%** of bankroll per bet
- **5%** maximum daily exposure, scaled back proportionally if breached
- no recommendation before official lineups are posted
- no recommendation when any required input is missing

The gate checks data completeness *before* it looks at the edge, so a missing
input can never quietly become a small edge.

## Layout

```
src/mlbline/    devig · odds · market · rundist · policy
scripts/        scan_market.py
tests/          test_model.py   (25 tests: de-vig identities, sim
                                 calibration, policy refusals)
data/snapshots/ raw odds payloads, timestamped
reports/        generated markdown
```

This tool produces analysis only. It never places a bet.
