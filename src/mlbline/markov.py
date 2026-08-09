"""Plate-appearance level baseball simulation over base-out states.

This replaces the independent half-inning draw in `rundist`. The motivation
is narrow and specific: the old engine had to *impose* a walk-off rule, since
it generated a whole half-inning's runs in one draw and then truncated the
result to a one-run win. That truncation piled probability mass at exactly
one run and produced 31.0% one-run games against MLB's ~28.5% -- an excess
sitting almost entirely in that single bucket, which biased every favourite's
-1.5 downward by roughly 2.8pp.

Simulating plate appearances removes the need for the rule. Play simply stops
when the winning run crosses, so a walk-off single scores one and a walk-off
grand slam scores four, in whatever proportion the batting events themselves
produce. The margin distribution becomes an output rather than an input.

Structure: eight base states crossed with three out counts, advanced by seven
plate-appearance outcomes. Baserunner advancement is probabilistic where real
baseball is (first-to-third on a single, scoring from first on a double,
double plays, sacrifice flies) and deterministic where it is forced.

Speed note: regulation innings are sampled from a precomputed run
distribution built by this same chain, which is exact for their purpose and
far faster. Only the bottom of the ninth and extra innings need
plate-by-plate detail, because only they can stop mid-inning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Plate appearance outcome codes.
K, BB, SINGLE, DOUBLE, TRIPLE, HR, OUT_IN_PLAY = range(7)

# League-average rates per plate appearance. On-base events sum to .316,
# matching a league OBP near .315.
LEAGUE_RATES = np.array([
    0.225,   # strikeout
    0.093,   # walk or hit by pitch
    0.142,   # single
    0.045,   # double
    0.004,   # triple
    0.032,   # home run
    0.459,   # out in play
])

# Baserunner advancement probabilities.
P_SCORE_FROM_SECOND_ON_SINGLE = 0.60
P_FIRST_TO_THIRD_ON_SINGLE = 0.28
P_SCORE_FROM_FIRST_ON_DOUBLE = 0.45
P_DOUBLE_PLAY = 0.13          # given a ball in play, runner on first, <2 outs
P_SAC_FLY = 0.16              # given a ball in play, runner on third, <2 outs

MAX_PA_PER_HALF_INNING = 40
MAX_EXTRA_INNINGS = 12
RUN_PMF_SAMPLES = 150_000
MAX_RUNS_IN_HALF = 20


def _play_half_innings(rng: np.random.Generator, rates: np.ndarray, n: int,
                       *, ghost_runner: bool = False,
                       stop_above: np.ndarray | None = None) -> np.ndarray:
    """Simulate n half-innings and return runs scored in each.

    `stop_above` ends a half-inning as soon as its run total exceeds the given
    value, which is how a walk-off works: the runs from the deciding play all
    count, so the final margin depends on what the batter actually did.
    """
    on1 = np.zeros(n, dtype=bool)
    on2 = np.full(n, ghost_runner, dtype=bool)
    on3 = np.zeros(n, dtype=bool)
    outs = np.zeros(n, dtype=np.int16)
    runs = np.zeros(n, dtype=np.int16)
    live = np.ones(n, dtype=bool)

    cumulative = np.cumsum(rates)

    for _ in range(MAX_PA_PER_HALF_INNING):
        if not live.any():
            break

        outcome = np.searchsorted(cumulative, rng.random(n))
        adv = rng.random((4, n))

        is_k = outcome == K
        is_bb = outcome == BB
        is_1b = outcome == SINGLE
        is_2b = outcome == DOUBLE
        is_3b = outcome == TRIPLE
        is_hr = outcome == HR
        is_out = outcome == OUT_IN_PLAY

        new1, new2, new3 = on1.copy(), on2.copy(), on3.copy()
        scored = np.zeros(n, dtype=np.int16)
        made_outs = np.zeros(n, dtype=np.int16)

        # Home run: everyone scores, bases clear.
        scored = np.where(is_hr, 1 + on1 + on2 + on3, scored)
        new1 = np.where(is_hr, False, new1)
        new2 = np.where(is_hr, False, new2)
        new3 = np.where(is_hr, False, new3)

        # Triple: all runners score, batter to third.
        scored = np.where(is_3b, on1 + on2 + on3, scored)
        new1 = np.where(is_3b, False, new1)
        new2 = np.where(is_3b, False, new2)
        new3 = np.where(is_3b, True, new3)

        # Double: runner on first scores some of the time.
        first_scores = on1 & (adv[0] < P_SCORE_FROM_FIRST_ON_DOUBLE)
        scored = np.where(is_2b, on3 + on2 + first_scores, scored)
        new1 = np.where(is_2b, False, new1)
        new2 = np.where(is_2b, True, new2)
        new3 = np.where(is_2b, on1 & ~first_scores, new3)

        # Single: third always scores; second usually does; first sometimes
        # takes third, but only when the runner ahead vacated it.
        second_scores = on2 & (adv[1] < P_SCORE_FROM_SECOND_ON_SINGLE)
        second_holds = on2 & ~second_scores
        first_to_third = on1 & (adv[2] < P_FIRST_TO_THIRD_ON_SINGLE) & ~second_holds
        scored = np.where(is_1b, on3 + second_scores, scored)
        new3 = np.where(is_1b, second_holds | first_to_third, new3)
        new2 = np.where(is_1b, on1 & ~first_to_third, new2)
        new1 = np.where(is_1b, True, new1)

        # Walk: forced advancement only, so a run scores only with the bases
        # loaded.
        scored = np.where(is_bb, (on1 & on2 & on3).astype(np.int16), scored)
        new3 = np.where(is_bb, on3 | (on1 & on2), new3)
        new2 = np.where(is_bb, on2 | on1, new2)
        new1 = np.where(is_bb, True, new1)

        # Strikeout.
        made_outs = np.where(is_k, 1, made_outs)

        # Ball in play for an out: possible double play, else possible
        # sacrifice fly, else runners hold.
        can_double = is_out & on1 & (outs < 2)
        double_play = can_double & (adv[3] < P_DOUBLE_PLAY)
        can_sac = is_out & ~double_play & on3 & (outs < 2)
        sac_fly = can_sac & (adv[3] < P_SAC_FLY)
        plain_out = is_out & ~double_play & ~sac_fly

        made_outs = np.where(double_play, 2, made_outs)
        made_outs = np.where(sac_fly | plain_out, 1, made_outs)
        new1 = np.where(double_play, False, new1)
        new3 = np.where(sac_fly, False, new3)
        scored = np.where(sac_fly, 1, scored)

        # Commit only for half-innings still in progress.
        on1 = np.where(live, new1, on1)
        on2 = np.where(live, new2, on2)
        on3 = np.where(live, new3, on3)
        runs = np.where(live, runs + scored, runs)
        outs = np.where(live, outs + made_outs, outs)

        live &= outs < 3
        if stop_above is not None:
            live &= runs <= stop_above

    return runs


def _run_pmf(rates: np.ndarray, seed: int) -> np.ndarray:
    """Empirical distribution of runs in a full half-inning."""
    rng = np.random.default_rng(seed)
    runs = _play_half_innings(rng, rates, RUN_PMF_SAMPLES)
    counts = np.bincount(np.clip(runs, 0, MAX_RUNS_IN_HALF),
                         minlength=MAX_RUNS_IN_HALF + 1)
    return counts / counts.sum()


def runs_per_nine(rates: np.ndarray, seed: int = 1) -> float:
    """Expected runs over nine innings for an offence with these rates."""
    return float(_run_pmf(rates, seed) @ np.arange(MAX_RUNS_IN_HALF + 1) * 9.0)


ON_BASE = np.array([False, True, True, True, True, True, False])

# Built once, then inverted by interpolation. Bisecting per call meant a
# fresh set of half-inning simulations for every candidate scoring level,
# which made solving a slate of games impractically slow.
_CALIBRATION: tuple[np.ndarray, np.ndarray] | None = None


def _calibration_curve() -> tuple[np.ndarray, np.ndarray]:
    """Monotone map from scale factor to runs per nine."""
    global _CALIBRATION
    if _CALIBRATION is None:
        factors = np.linspace(0.35, 2.10, 26)
        runs = np.array([runs_per_nine(_apply_scale(f, ON_BASE), seed=3)
                         for f in factors])
        order = np.argsort(runs)
        _CALIBRATION = (runs[order], factors[order])
    return _CALIBRATION


def scale_rates(target_runs: float) -> np.ndarray:
    """Plate-appearance rates producing a given runs-per-nine.

    On-base events scale together and the outs absorb the remainder, keeping
    the shape of the offence while moving its level. Note the resulting rate
    vector is not a literal scouting line -- its on-base figure runs high
    because this chain has no productive outs, steals or errors, and the
    scaling compensates so that the *run distribution* matches. Runs are what
    get priced; the peripherals are a means to them.
    """
    runs, factors = _calibration_curve()
    factor = float(np.interp(target_runs, runs, factors))
    return _apply_scale(factor, ON_BASE)


def _apply_scale(factor: float, on_base: np.ndarray) -> np.ndarray:
    rates = LEAGUE_RATES.copy()
    rates[on_base] *= factor
    reached = rates[on_base].sum()
    if reached >= 0.95:                       # keep the inning terminating
        rates[on_base] *= 0.95 / reached
        reached = 0.95
    outs = LEAGUE_RATES[~on_base]
    rates[~on_base] = outs / outs.sum() * (1.0 - reached)
    return rates


@dataclass
class MarkovSimulation:
    """Same surface as `rundist.GameSimulation`, different engine."""

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
        return self.home_runs - self.away_runs

    def prob_home_win(self) -> float:
        return float(np.mean(self.margin > 0))

    def prob_cover(self, point: float, *, home: bool) -> float:
        margin = self.margin if home else -self.margin
        pushes = np.sum(margin == -point)
        live = self.n - pushes
        return float(np.sum(margin > -point) / live) if live else 0.0

    def prob_over(self, line: float) -> float:
        pushes = np.sum(self.total == line)
        live = self.n - pushes
        return float(np.sum(self.total > line) / live) if live else 0.0

    def prob_under(self, line: float) -> float:
        pushes = np.sum(self.total == line)
        live = self.n - pushes
        return float(np.sum(self.total < line) / live) if live else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "mean_home": float(np.mean(self.home_runs)),
            "mean_away": float(np.mean(self.away_runs)),
            "mean_total": float(np.mean(self.total)),
            "mean_margin": float(np.mean(self.margin)),
            "sd_total": float(np.std(self.total)),
            "home_win": self.prob_home_win(),
        }


_RATE_CACHE: dict[float, np.ndarray] = {}
_PMF_CACHE: dict[bytes, np.ndarray] = {}


def _cached_rates(target: float) -> np.ndarray:
    # Snapped to a 0.02-run grid. The solver probes many nearby scoring
    # levels and each distinct one costs a fresh half-inning distribution;
    # 0.02 runs is far finer than the ~0.35 the pricing can actually resolve,
    # so the grid costs nothing real and turns most probes into cache hits.
    key = round(target / 0.02) * 0.02
    if key not in _RATE_CACHE:
        _RATE_CACHE[key] = scale_rates(key)
    return _RATE_CACHE[key]


def _cached_pmf(rates: np.ndarray) -> np.ndarray:
    key = rates.tobytes()
    if key not in _PMF_CACHE:
        _PMF_CACHE[key] = _run_pmf(rates, seed=7)
    return _PMF_CACHE[key]


def simulate_game(expected_home_runs: float, expected_away_runs: float, *,
                  n_sims: int = 200_000, seed: int | None = None,
                  extras_multiplier: float = 1.0) -> MarkovSimulation:
    """Simulate a game from each side's expected runs per nine innings.

    `extras_multiplier` is retained for interface compatibility but is not
    used: extra innings get the automatic runner on second, which raises
    scoring through the chain itself rather than through a fudge factor.
    """
    del extras_multiplier

    rng = np.random.default_rng(seed)
    home_rates = _cached_rates(expected_home_runs)
    away_rates = _cached_rates(expected_away_runs)
    home_cdf = np.cumsum(_cached_pmf(home_rates))
    away_cdf = np.cumsum(_cached_pmf(away_rates))

    # Regulation: away bats nine times, home eight for certain. Inverse-CDF
    # sampling rather than rng.choice, which is orders of magnitude slower
    # with an explicit probability vector and dominated the solver's runtime.
    away_runs = np.searchsorted(
        away_cdf, rng.random((n_sims, 9))).sum(axis=1).astype(np.int32)
    home_runs = np.searchsorted(
        home_cdf, rng.random((n_sims, 8))).sum(axis=1).astype(np.int32)

    # Bottom of the ninth, played only when the home team is not ahead and
    # ended by the go-ahead run rather than by a truncation rule. Only the
    # games that actually reach it are simulated.
    bats = home_runs <= away_runs
    batting = np.flatnonzero(bats)
    if batting.size:
        deficit = away_runs[batting] - home_runs[batting]
        ninth = _play_half_innings(rng, home_rates, batting.size,
                                   stop_above=deficit)
        home_runs[batting] += ninth

    home_runs, away_runs = _play_extras(rng, home_runs, away_runs,
                                        home_rates, away_rates)
    return MarkovSimulation(home_runs=home_runs, away_runs=away_runs)


def _play_extras(rng: np.random.Generator,
                 home_runs: np.ndarray, away_runs: np.ndarray,
                 home_rates: np.ndarray, away_rates: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ties, with the automatic runner starting on second."""
    for _ in range(MAX_EXTRA_INNINGS):
        tied = home_runs == away_runs
        count = int(np.sum(tied))
        if count == 0:
            break

        top = _play_half_innings(rng, away_rates, count, ghost_runner=True)
        away_after = away_runs[tied] + top

        deficit = np.maximum(away_after - home_runs[tied], 0)
        bottom = _play_half_innings(rng, home_rates, count,
                                    ghost_runner=True, stop_above=deficit)

        away_runs[tied] = away_after
        home_runs[tied] = home_runs[tied] + bottom
    return home_runs, away_runs
