"""The staking rules, written down so a recommendation can be audited.

Encoding the thresholds as code rather than prose matters because the failure
mode of a betting model is not a wrong number, it is a plausible-looking
number produced from inputs that were never there. `evaluate` refuses on
missing inputs before it ever looks at the edge, so "we had no pitcher data"
cannot quietly become "we found a small edge".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BettingPolicy:
    """Thresholds a candidate must clear before it can be recommended."""

    min_expected_value: float = 0.04       # +4% on the model's own numbers
    min_probability_divergence: float = 0.03   # 3 points vs de-vigged market
    kelly_multiplier: float = 0.25
    max_single_stake: float = 0.01         # 1% of bankroll
    max_daily_exposure: float = 0.05       # 5% of bankroll
    require_confirmed_lineups: bool = True
    min_books_for_consensus: int = 5


@dataclass
class DataCompleteness:
    """Which handicapping inputs were actually obtained for a game."""

    starting_pitchers: bool = False
    confirmed_lineups: bool = False
    bullpen_usage: bool = False
    team_offence_metrics: bool = False
    park_factors: bool = False
    weather: bool = False
    market_prices: bool = False

    REQUIRED = (
        "starting_pitchers",
        "team_offence_metrics",
        "bullpen_usage",
        "park_factors",
        "weather",
        "market_prices",
    )

    def missing(self) -> list[str]:
        return [f for f in self.REQUIRED if not getattr(self, f)]

    def sufficient_for_model(self) -> bool:
        return not self.missing()


@dataclass
class GateResult:
    """Outcome of applying the policy, with every reason it failed."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    stake_fraction: float = 0.0

    def __bool__(self) -> bool:
        return self.passed


def evaluate(*, policy: BettingPolicy,
             completeness: DataCompleteness,
             model_probability: float | None,
             market_probability: float | None,
             decimal_odds: float | None,
             n_books: int = 0) -> GateResult:
    """Decide whether a candidate may be recommended.

    `model_probability` must come from an independent estimate of the run
    distribution. Passing the market's own de-vigged probability back in is
    circular and is rejected: it would always show zero divergence anyway,
    but making it explicit keeps the failure loud.
    """
    reasons: list[str] = []

    missing = completeness.missing()
    if missing:
        reasons.append(
            "insufficient data: missing " + ", ".join(missing)
        )

    if policy.require_confirmed_lineups and not completeness.confirmed_lineups:
        reasons.append("official lineups not yet posted; observation only")

    if model_probability is None:
        reasons.append(
            "no model probability: run distribution could not be estimated"
        )

    if market_probability is None:
        reasons.append("no de-vigged market probability available")

    if n_books < policy.min_books_for_consensus:
        reasons.append(
            f"thin market: {n_books} books, need {policy.min_books_for_consensus}"
        )

    if reasons:
        return GateResult(passed=False, reasons=reasons)

    assert model_probability is not None
    assert market_probability is not None
    assert decimal_odds is not None

    divergence = model_probability - market_probability
    expected_value = model_probability * decimal_odds - 1.0

    if divergence < policy.min_probability_divergence:
        reasons.append(
            f"divergence {divergence * 100:.2f}pp below "
            f"{policy.min_probability_divergence * 100:.0f}pp minimum"
        )
    if expected_value < policy.min_expected_value:
        reasons.append(
            f"expected value {expected_value * 100:.2f}% below "
            f"{policy.min_expected_value * 100:.0f}% minimum"
        )

    if reasons:
        return GateResult(passed=False, reasons=reasons)

    b = decimal_odds - 1.0
    kelly = (model_probability * b - (1.0 - model_probability)) / b
    stake = min(policy.kelly_multiplier * kelly, policy.max_single_stake)
    return GateResult(passed=True, reasons=[], stake_fraction=stake)


def cap_daily_exposure(stakes: list[float],
                       policy: BettingPolicy) -> list[float]:
    """Scale stakes down proportionally if the day's total breaches the cap."""
    total = sum(stakes)
    if total <= policy.max_daily_exposure or total == 0:
        return list(stakes)
    scale = policy.max_daily_exposure / total
    return [s * scale for s in stakes]
