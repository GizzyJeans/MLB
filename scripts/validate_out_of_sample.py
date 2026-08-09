#!/usr/bin/env python3
"""Test the simulator against real game results it was never tuned on.

  python3 scripts/validate_out_of_sample.py --games data/history/games.csv

Design matters here, because a model can be made to match almost any single
statistic by tuning. So the fit and the test are kept separate:

  fitted   each side's mean and variance of runs per game, by choosing the
           average expected runs for home and away and the spread of matchup
           strengths across the schedule. Four numbers.

  tested   the margin distribution, the one-run game rate, and the share of
           wins that are by two or more runs, separately for home and away.

The marginals do not determine the tested quantities. Two teams' run
distributions can be exactly right while the margin distribution is wrong,
which is precisely what happens when the bottom of the ninth and walk-offs
are handled by a truncation rule instead of being played out. That is the
gap this measures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlbline import markov, rundist  # noqa: E402

LAMBDA_GRID = 0.1
N_MATCHUPS = 400
SIMS_PER_MATCHUP = 3000


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    home = np.array([int(r["home_runs"]) for r in rows])
    away = np.array([int(r["away_runs"]) for r in rows])
    return home, away


def statistics(home: np.ndarray, away: np.ndarray) -> dict[str, float]:
    margin = home - away
    abs_margin = np.abs(margin)
    won_home = margin > 0
    won_away = margin < 0
    return {
        "mean_home": float(np.mean(home)),
        "mean_away": float(np.mean(away)),
        "var_home": float(np.var(home)),
        "var_away": float(np.var(away)),
        "home_win": float(np.mean(won_home)),
        "one_run": float(np.mean(abs_margin == 1)),
        "two_run": float(np.mean(abs_margin == 2)),
        "three_run": float(np.mean(abs_margin == 3)),
        "four_run": float(np.mean(abs_margin == 4)),
        "five_run": float(np.mean(abs_margin == 5)),
        "six_plus": float(np.mean(abs_margin >= 6)),
        "home_by_two_share": float(np.mean(margin >= 2) / np.mean(won_home)),
        "away_by_two_share": float(np.mean(margin <= -2) / np.mean(won_away)),
        "mean_total": float(np.mean(home + away)),
    }


def simulate_league(engine, mu_home: float, mu_away: float, spread: float,
                    seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a schedule with matchup strengths drawn around the means."""
    rng = np.random.default_rng(seed)
    lam_home = np.clip(rng.normal(mu_home, spread, N_MATCHUPS), 2.2, 8.0)
    lam_away = np.clip(rng.normal(mu_away, spread, N_MATCHUPS), 2.2, 8.0)
    lam_home = np.round(lam_home / LAMBDA_GRID) * LAMBDA_GRID
    lam_away = np.round(lam_away / LAMBDA_GRID) * LAMBDA_GRID

    homes, aways = [], []
    for i in range(N_MATCHUPS):
        sim = engine.simulate_game(float(lam_home[i]), float(lam_away[i]),
                                   n_sims=SIMS_PER_MATCHUP, seed=seed + i)
        homes.append(sim.home_runs)
        aways.append(sim.away_runs)
    return np.concatenate(homes), np.concatenate(aways)


def fit(engine, target: dict[str, float], seed: int = 4
        ) -> tuple[float, float, float]:
    """Choose mean expected runs per side and the spread of matchup strength.

    Only the four marginal moments are used. Nothing about margins enters.
    """
    best = None
    for spread in np.arange(0.40, 1.45, 0.15):
        mu_home, mu_away = target["mean_home"], target["mean_away"]
        # Mean runs is monotone in the input, so a few passes suffice.
        for _ in range(4):
            home, away = simulate_league(engine, mu_home, mu_away, spread, seed)
            mu_home += target["mean_home"] - float(np.mean(home))
            mu_away += target["mean_away"] - float(np.mean(away))
        home, away = simulate_league(engine, mu_home, mu_away, spread, seed)
        got = statistics(home, away)
        error = (abs(got["var_home"] - target["var_home"])
                 + abs(got["var_away"] - target["var_away"]))
        if best is None or error < best[0]:
            best = (error, mu_home, mu_away, spread)
    _, mu_home, mu_away, spread = best
    return mu_home, mu_away, spread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="data/history/games.csv")
    args = parser.parse_args()

    home, away = load(args.games)
    target = statistics(home, away)
    print(f"實證樣本: {len(home)} 場\n")

    results = {}
    for name, engine in (("markov", markov), ("rundist", rundist)):
        mu_home, mu_away, spread = fit(engine, target)
        sim_home, sim_away = simulate_league(engine, mu_home, mu_away, spread)
        results[name] = statistics(sim_home, sim_away)
        results[name]["_fit"] = (mu_home, mu_away, spread)
        print(f"{name}: 擬合 mu_home={mu_home:.2f} mu_away={mu_away:.2f} "
              f"spread={spread:.2f}")

    fitted = ["mean_home", "mean_away", "var_home", "var_away"]
    tested = ["home_win", "one_run", "two_run", "three_run", "four_run",
              "five_run", "six_plus", "home_by_two_share",
              "away_by_two_share", "mean_total"]

    print(f"\n{'':22s} {'實證':>9s} {'markov':>9s} {'rundist':>9s}")
    print("--- 擬合的量 (marginals) " + "-" * 30)
    for key in fitted:
        print(f"{key:22s} {target[key]:9.3f} "
              f"{results['markov'][key]:9.3f} {results['rundist'][key]:9.3f}")

    print("--- 測試的量 (out-of-sample) " + "-" * 26)
    errors = {"markov": 0.0, "rundist": 0.0}
    for key in tested:
        scale = 100.0 if key != "mean_total" else 1.0
        line = f"{key:22s} {target[key] * scale:9.2f}"
        for name in ("markov", "rundist"):
            value = results[name][key]
            line += f" {value * scale:9.2f}"
            if key != "mean_total":
                errors[name] += abs(value - target[key]) * 100
        print(line)

    print(f"\n分差相關統計的總絕對誤差:")
    print(f"  markov  {errors['markov']:.2f}pp")
    print(f"  rundist {errors['rundist']:.2f}pp")
    winner = min(errors, key=errors.get)
    print(f"  => {winner} 較佳，差距 "
          f"{abs(errors['markov'] - errors['rundist']):.2f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
