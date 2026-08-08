"""Monte Carlo run distribution for a single MLB game.

The point of simulating half-innings rather than drawing two game totals from
a pair of Poissons is that three structural rules move run-line and total
prices by more than most handicapping edges are worth:

1. The home team does not bat in the bottom of the 9th when it already leads.
   This removes roughly a fifth of a run from the expected total and, more
   importantly, truncates the home team's winning margin -- which is exactly
   what a -1.5 run line is priced on. A symmetric model will systematically
   overprice home favourites on the run line.

2. A walk-off ends play the moment the run scores, capping the home margin at
   one run unless the winning hit clears the fence.

3. Extra innings start with a runner on second, so scoring per half-inning in
   extras is well above the regulation rate, fattening the over tail.

Run scoring per half-inning is modelled as a Gamma-Poisson (negative
binomial) mixture. Pure Poisson badly understates the variance of team run
totals: real MLB team scores have a variance-to-mean ratio near 2.1, not 1.0,
because innings are not independent Bernoulli events -- big innings cluster.
`HALF_INNING_DISPERSION` is calibrated to reproduce that ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Gamma shape for the per-half-inning Gamma-Poisson mixture. Calibrated so a
# team averaging 4.5 runs per nine innings shows a game-score variance near
# 9.7 (variance/mean ~ 2.15), matching league-wide team scoring.
HALF_INNING_DISPERSION = 0.43

# Scoring multiplier for extra-inning half-innings, reflecting the automatic
# runner placed on second base.
EXTRA_INNING_SCORING_MULTIPLIER = 1.35

# Share of walk-off wins that clear the fence and so score more than the
# single run needed. Sets how often a home win exceeds a one-run margin in
# the bottom half.
WALKOFF_OVERSHOOT_RATE = 0.20

MAX_EXTRA_INNINGS = 12

# KNOWN CALIBRATION TENSION -- read before trusting a run-line edge.
#
# Two league-wide anchors cannot both be hit by this model structure:
#
#   dispersion 0.43 -> variance/mean 2.14 (real 2.15 ✓) but 31.0% one-run
#                      games (real ~28.5% ✗)
#   dispersion 0.35 -> 29.9% one-run games (closer ✓) but variance/mean 2.41 ✗
#
# The cause is structural: innings are drawn independently, and the two
# teams' scores are uncorrelated. Real games have neither property -- the
# same pitcher works several innings, lineups turn over, and bullpen usage
# responds to the score. So the margin distribution comes out too tight even
# when the score distribution is right.
#
# Consequence: with the current settings the model produces about 2.5pp too
# many one-run games, which biases every favourite's -1.5 downward by
# roughly 2.8pp and every underdog's +1.5 upward by the same amount. Across
# the plausible parameter range P(home -1.5) at pick-em moves over a ~3pp
# band (29.8%-32.8%).
#
# Any run-line edge under roughly 3pp is therefore inside this model's own
# specification uncertainty and must not be staked. Fixing it properly needs
# a base-out state Markov chain rather than independent half-innings.


@dataclass
class GameSimulation:
    """Simulated score pairs and the quantities priced off them."""

    home_runs: np.ndarray
    away_runs: np.ndarray

    @property
    def n(self) -> int:
        return len(self.home_runs)

    @property
    def total(self) -> np.ndarray:
        return self.home_runs + self.away_runs

    @property
    def margin(self) -> np.ndarray:
        """Home margin. Positive means the home team won."""
        return self.home_runs - self.away_runs

    def prob_home_win(self) -> float:
        return float(np.mean(self.margin > 0))

    def prob_cover(self, point: float, *, home: bool) -> float:
        """Probability a run line cashes.

        `point` follows betting convention: -1.5 lays the runs, +1.5 takes
        them. Half-run lines cannot push; integer lines are handled by
        excluding pushes from the denominator, which is how books settle them.
        """
        margin = self.margin if home else -self.margin
        wins = np.sum(margin > -point)
        pushes = np.sum(margin == -point)
        live = self.n - pushes
        return float(wins / live) if live else 0.0

    def prob_over(self, line: float) -> float:
        total = self.total
        overs = np.sum(total > line)
        pushes = np.sum(total == line)
        live = self.n - pushes
        return float(overs / live) if live else 0.0

    def prob_under(self, line: float) -> float:
        total = self.total
        unders = np.sum(total < line)
        pushes = np.sum(total == line)
        live = self.n - pushes
        return float(unders / live) if live else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "mean_home": float(np.mean(self.home_runs)),
            "mean_away": float(np.mean(self.away_runs)),
            "mean_total": float(np.mean(self.total)),
            "median_total": float(np.median(self.total)),
            "mean_margin": float(np.mean(self.margin)),
            "sd_total": float(np.std(self.total)),
            "home_win": self.prob_home_win(),
        }


def _draw_half_inning_runs(rng: np.random.Generator, mean: np.ndarray | float,
                           size: tuple[int, ...],
                           dispersion: float = HALF_INNING_DISPERSION
                           ) -> np.ndarray:
    """Gamma-Poisson draw of runs in a half-inning."""
    mean_arr = np.broadcast_to(np.asarray(mean, dtype=float), size)
    # Gamma with shape r and scale mean/r has mean `mean` and adds the
    # overdispersion; Poisson conditional on it yields negative binomial.
    rates = rng.gamma(shape=dispersion, scale=mean_arr / dispersion, size=size)
    return rng.poisson(rates)


def simulate_game(expected_home_runs: float, expected_away_runs: float, *,
                  n_sims: int = 200_000,
                  seed: int | None = None,
                  dispersion: float = HALF_INNING_DISPERSION,
                  extras_multiplier: float = EXTRA_INNING_SCORING_MULTIPLIER,
                  walkoff_overshoot: float = WALKOFF_OVERSHOOT_RATE,
                  ) -> GameSimulation:
    """Simulate a game from each side's expected runs per nine innings.

    `expected_home_runs` / `expected_away_runs` are the handicapping inputs:
    the runs each offence is expected to score against the specific opposing
    pitching staff, in this park, in these conditions. Producing them is the
    hard part and is not this function's job.
    """
    if expected_home_runs <= 0 or expected_away_runs <= 0:
        raise ValueError("expected runs must be positive")

    rng = np.random.default_rng(seed)
    home_rate = expected_home_runs / 9.0
    away_rate = expected_away_runs / 9.0

    # Regulation: away bats nine times, home bats eight times for certain.
    away_runs = _draw_half_inning_runs(
        rng, away_rate, (n_sims, 9), dispersion).sum(axis=1)
    home_runs = _draw_half_inning_runs(
        rng, home_rate, (n_sims, 8), dispersion).sum(axis=1)

    # Bottom of the 9th is played only when the home team is not ahead.
    bats_ninth = home_runs <= away_runs
    ninth = _draw_half_inning_runs(rng, home_rate, (n_sims,), dispersion)
    ninth = np.where(bats_ninth, ninth, 0)
    home_runs = home_runs + ninth

    home_runs, away_runs = _apply_walkoff_truncation(
        rng, home_runs, away_runs, bats_ninth, walkoff_overshoot)

    home_runs, away_runs = _play_extra_innings(
        rng, home_runs, away_runs, home_rate * extras_multiplier,
        away_rate * extras_multiplier, dispersion, walkoff_overshoot)

    return GameSimulation(home_runs=home_runs, away_runs=away_runs)


def _apply_walkoff_truncation(rng: np.random.Generator,
                              home_runs: np.ndarray, away_runs: np.ndarray,
                              batted: np.ndarray,
                              overshoot_rate: float
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Cap a home team's winning margin in a half-inning it walked off in.

    Play stops on the go-ahead run, so the margin is one -- unless the hit was
    a home run, which is what `overshoot_rate` allows for.
    """
    walked_off = batted & (home_runs > away_runs)
    if not np.any(walked_off):
        return home_runs, away_runs

    capped = away_runs + 1
    overshoot = rng.random(len(home_runs)) < overshoot_rate
    # A walk-off home run scores the runners already aboard: 2 to 4 runs.
    extra = rng.integers(1, 4, size=len(home_runs))
    capped_with_hr = np.where(overshoot, capped + extra, capped)

    home_runs = np.where(walked_off,
                         np.minimum(home_runs, capped_with_hr),
                         home_runs)
    return home_runs, away_runs


def _play_extra_innings(rng: np.random.Generator,
                        home_runs: np.ndarray, away_runs: np.ndarray,
                        home_rate: float, away_rate: float,
                        dispersion: float,
                        overshoot_rate: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Resolve tied games, one extra inning at a time."""
    for _ in range(MAX_EXTRA_INNINGS):
        tied = home_runs == away_runs
        if not np.any(tied):
            break
        n_tied = int(np.sum(tied))

        top = _draw_half_inning_runs(rng, away_rate, (n_tied,), dispersion)
        bottom = _draw_half_inning_runs(rng, home_rate, (n_tied,), dispersion)

        away_after = away_runs[tied] + top
        home_before = home_runs[tied]

        # The home team bats unless the visitors' half already decided it --
        # which it cannot, since the score was level; so it always bats, but
        # stops the moment it goes ahead.
        home_after = home_before + bottom
        walked_off = home_after > away_after
        capped = away_after + 1
        overshoot = rng.random(n_tied) < overshoot_rate
        extra = rng.integers(1, 4, size=n_tied)
        capped_with_hr = np.where(overshoot, capped + extra, capped)
        home_after = np.where(walked_off,
                              np.minimum(home_after, capped_with_hr),
                              home_after)

        away_runs[tied] = away_after
        home_runs[tied] = home_after

    return home_runs, away_runs


def implied_expected_runs(total_line: float, home_win_prob: float,
                          *, tolerance: float = 0.02,
                          max_iter: int = 60) -> tuple[float, float]:
    """Back out each side's expected runs from a total and a win probability.

    Useful for sanity-checking a handicap against the market, or for seeding a
    model when only market prices are available. Note that recovering inputs
    from prices and then "testing" those inputs against the same prices is
    circular -- it can validate internal consistency, never edge.
    """
    lo, hi = 0.05, 0.95
    for _ in range(max_iter):
        share = 0.5 * (lo + hi)
        home = total_line * share
        away = total_line * (1.0 - share)
        sim = simulate_game(home, away, n_sims=40_000, seed=7)
        if abs(sim.prob_home_win() - home_win_prob) < tolerance:
            return home, away
        if sim.prob_home_win() < home_win_prob:
            lo = share
        else:
            hi = share
    share = 0.5 * (lo + hi)
    return total_line * share, total_line * (1.0 - share)
