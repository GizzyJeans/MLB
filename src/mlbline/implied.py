"""Price the run line from the moneyline and the total.

The moneyline says who wins; the total says how much scoring there is.
Together they pin down each side's expected runs, and once those are known
the run line is no longer a free parameter -- it is a derivative, and the
simulation in `rundist` prices it. Where the quoted run line disagrees with
that derived price, one of the three markets is out of step with the other
two.

This is relative value, not handicapping. It finds run lines that have
drifted away from the moneyline and total; it cannot find a game the whole
market has mispriced, because it takes the moneyline and total as given. The
distinction matters when sizing: an edge measured against other markets
assumes those markets are right.

The solver alternates two one-dimensional searches, which converges quickly
because the coupling is weak -- the total is governed almost entirely by the
scoring level and the moneyline almost entirely by how that scoring splits.
Every evaluation reuses the same random seed so the objective is
deterministic and the bisection does not chase Monte Carlo noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .markov import simulate_game

SOLVE_SIMS = 60_000
PRICE_SIMS = 400_000
SOLVE_SEED = 20260808


@dataclass
class ImpliedPricing:
    """Expected runs implied by the moneyline and total, and what they price."""

    matchup: str
    total_line: float
    market_home_win: float
    market_over: float
    lam_home: float
    lam_away: float
    fitted_home_win: float
    fitted_over: float
    p_home_cover: float      # home -1.5
    p_away_cover: float      # away +1.5
    mean_total: float
    mean_margin: float
    # Kept so callers can price whichever run line the market actually hangs,
    # not just the home -1.5 case.
    simulation: object = None
    # (line, fitted - market) for every total the market quotes.
    over_residuals: tuple[tuple[float, float], ...] = ()

    @property
    def fit_error(self) -> float:
        """Worst residual against the two anchor markets, in probability."""
        return max(abs(self.fitted_home_win - self.market_home_win),
                   abs(self.fitted_over - self.market_over))

    @property
    def anchor_spread(self) -> float:
        """Worst residual across every quoted total, in probability.

        With one anchor the fit is exact by construction and says nothing.
        Across several it reports whether the market's own lines agree with
        each other under a single scoring level -- a wide spread means the
        quoted lines are mutually inconsistent, so any price extrapolated
        between them is less trustworthy than its EV suggests.
        """
        if not self.over_residuals:
            return 0.0
        return max(abs(r) for _, r in self.over_residuals)

    @property
    def predicted_score(self) -> str:
        return f"{self.lam_away:.1f} - {self.lam_home:.1f}"


def _prob_over(total: float, share: float, line: float, n_sims: int) -> float:
    sim = simulate_game(total * share, total * (1.0 - share),
                        n_sims=n_sims, seed=SOLVE_SEED)
    return sim.prob_over(line)


def _weighted_over_residual(total: float, share: float,
                            anchors: list[tuple[float, float, float]],
                            n_sims: int) -> float:
    """Book-weighted mean of (fitted - market) across every quoted total.

    Each term is increasing in the scoring level, so their weighted mean is
    too, which keeps the bisection valid while using all of the market's
    information instead of one line's worth.
    """
    sim = simulate_game(total * share, total * (1.0 - share),
                        n_sims=n_sims, seed=SOLVE_SEED)
    weight = sum(w for _, _, w in anchors)
    return sum(w * (sim.prob_over(line) - market)
               for line, market, w in anchors) / weight


def _prob_home_win(total: float, share: float, n_sims: int) -> float:
    sim = simulate_game(total * share, total * (1.0 - share),
                        n_sims=n_sims, seed=SOLVE_SEED)
    return sim.prob_home_win()


def _bisect(f, target, lo, hi, steps, increasing) -> float:
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        value = f(mid)
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve(matchup: str, *, market_home_win: float, market_over: float,
          total_line: float, outer: int = 4, steps: int = 14,
          n_sims: int = SOLVE_SIMS, price_sims: int = PRICE_SIMS,
          over_anchors: list[tuple[float, float, float]] | None = None,
          ) -> ImpliedPricing:
    """Recover expected runs per side, then price the run line off them.

    `over_anchors` is an optional list of (line, de-vigged P(over), weight)
    covering every total the market quotes. Fitting to one line makes the
    result depend on which line happens to carry the most books: when the
    Rays' modal total migrated from 7.5 to 7.0 without the market moving at
    all -- 7.0 read 0.5217 in both snapshots -- the fitted total shifted
    0.14 runs and moved a candidate's EV by 3.6 points. Weighting every
    quoted line by its book count removes that dependence.
    """
    total = total_line
    share = 0.5
    anchors = over_anchors or [(total_line, market_over, 1.0)]

    for _ in range(outer):
        # Scoring level: more expected runs means more overs at a fixed line.
        total = _bisect(
            lambda t: _weighted_over_residual(t, share, anchors, n_sims),
            0.0, 3.0, 16.0, steps, increasing=True)
        # Split: giving the home side a bigger share raises its win chance.
        share = _bisect(
            lambda s: _prob_home_win(total, s, n_sims),
            market_home_win, 0.25, 0.75, steps, increasing=True)

    lam_home = total * share
    lam_away = total * (1.0 - share)
    sim = simulate_game(lam_home, lam_away, n_sims=price_sims, seed=SOLVE_SEED + 1)
    summary = sim.summary()

    return ImpliedPricing(
        matchup=matchup,
        total_line=total_line,
        market_home_win=market_home_win,
        market_over=market_over,
        lam_home=lam_home,
        lam_away=lam_away,
        fitted_home_win=sim.prob_home_win(),
        fitted_over=sim.prob_over(total_line),
        p_home_cover=sim.prob_cover(-1.5, home=True),
        p_away_cover=sim.prob_cover(1.5, home=False),
        mean_total=summary["mean_total"],
        mean_margin=summary["mean_margin"],
        simulation=sim,
        over_residuals=tuple((line, sim.prob_over(line) - market)
                             for line, market, _ in anchors),
    )
