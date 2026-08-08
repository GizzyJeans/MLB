"""Checks on the de-vig estimators and the run-distribution engine.

The simulation assertions are calibration tests: they pin the engine to
published league-wide regularities, so a refactor that quietly breaks the
bottom-of-the-9th rule or the overdispersion fails loudly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from mlbline.devig import (  # noqa: E402
    american_to_decimal,
    decimal_to_american,
    devig_additive,
    devig_all,
    devig_multiplicative,
    devig_power,
    devig_shin,
    overround,
)
from mlbline.market import kelly_fraction  # noqa: E402
from mlbline.policy import (  # noqa: E402
    BettingPolicy,
    DataCompleteness,
    cap_daily_exposure,
    evaluate,
)
from mlbline.rundist import simulate_game  # noqa: E402


def _full_data() -> DataCompleteness:
    return DataCompleteness(
        starting_pitchers=True, confirmed_lineups=True, bullpen_usage=True,
        team_offence_metrics=True, park_factors=True, weather=True,
        market_prices=True,
    )


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


def test_odds_conversions_round_trip():
    for american in (-250, -150, -110, -101, 100, 110, 175, 400):
        dec = american_to_decimal(american)
        approx(decimal_to_american(dec), american, tol=1e-6)
    approx(american_to_decimal(100), 2.0)
    approx(american_to_decimal(-200), 1.5)


def test_overround_is_positive_on_standard_juice():
    dec = [american_to_decimal(-110), american_to_decimal(-110)]
    assert 0.04 < overround(dec) < 0.05


def test_all_devig_methods_sum_to_one():
    for pair in ([-110, -110], [-250, 210], [-1000, 650], [125, -145]):
        dec = [american_to_decimal(a) for a in pair]
        for name, probs in devig_all(dec).items():
            approx(sum(probs), 1.0, tol=1e-6)
            assert all(0.0 < p < 1.0 for p in probs), name


def test_methods_agree_on_balanced_prices():
    """On -110/-110 every estimator must return 0.5 -- symmetry forces it."""
    dec = [american_to_decimal(-110)] * 2
    for probs in devig_all(dec).values():
        approx(probs[0], 0.5, tol=1e-6)


def test_shin_shades_more_probability_off_the_longshot():
    """Shin loads vig onto the longshot, so it should sit below multiplicative."""
    dec = [american_to_decimal(-300), american_to_decimal(250)]
    longshot = 1
    mult = devig_multiplicative(dec)[longshot]
    shin = devig_shin(dec)[longshot]
    assert shin < mult, f"shin {shin} should be under multiplicative {mult}"


def test_additive_falls_back_when_it_would_go_negative():
    dec = [american_to_decimal(-100000), american_to_decimal(8000)]
    probs = devig_additive(dec)
    assert all(p > 0 for p in probs)


def test_power_method_normalises():
    dec = [american_to_decimal(-180), american_to_decimal(160)]
    approx(sum(devig_power(dec)), 1.0, tol=1e-6)


def test_home_team_scores_less_than_away_at_equal_strength():
    """The skipped bottom of the 9th is worth roughly a fifth of a run."""
    sim = simulate_game(4.5, 4.5, n_sims=120_000, seed=1)
    s = sim.summary()
    deficit = s["mean_away"] - s["mean_home"]
    assert 0.05 < deficit < 0.35, deficit


def test_win_probability_is_symmetric_at_equal_strength():
    """Equal offences give a near coin-flip; home edge here is small."""
    sim = simulate_game(4.5, 4.5, n_sims=200_000, seed=2)
    assert 0.47 < sim.prob_home_win() < 0.53, sim.prob_home_win()


def test_run_totals_are_overdispersed_relative_to_poisson():
    """Team scores must show variance well above their mean."""
    sim = simulate_game(4.5, 4.5, n_sims=200_000, seed=3)
    mean = float(np.mean(sim.home_runs))
    var = float(np.var(sim.home_runs))
    ratio = var / mean
    assert 1.6 < ratio < 2.8, f"variance/mean ratio {ratio}"


def test_scoring_responds_monotonically_to_inputs():
    low = simulate_game(3.5, 3.5, n_sims=80_000, seed=4).summary()
    high = simulate_game(5.5, 5.5, n_sims=80_000, seed=4).summary()
    assert high["mean_total"] > low["mean_total"] + 3.0


def test_run_line_and_moneyline_stay_consistent():
    """Laying 1.5 must always be less likely than winning outright."""
    sim = simulate_game(5.2, 4.0, n_sims=200_000, seed=5)
    assert sim.prob_cover(-1.5, home=True) < sim.prob_home_win()
    assert sim.prob_cover(1.5, home=False) > (1.0 - sim.prob_home_win())


def test_complementary_probabilities_close_on_half_lines():
    """Half-run lines cannot push, so the two sides must sum to one."""
    sim = simulate_game(4.8, 4.4, n_sims=150_000, seed=6)
    approx(sim.prob_cover(-1.5, home=True) + sim.prob_cover(1.5, home=False),
           1.0, tol=1e-6)
    approx(sim.prob_over(8.5) + sim.prob_under(8.5), 1.0, tol=1e-6)


def test_integer_totals_exclude_pushes_from_both_sides():
    sim = simulate_game(4.5, 4.5, n_sims=150_000, seed=8)
    total = sim.prob_over(9.0) + sim.prob_under(9.0)
    approx(total, 1.0, tol=1e-6)


def test_extra_innings_terminate():
    sim = simulate_game(4.5, 4.5, n_sims=50_000, seed=9)
    ties = int(np.sum(sim.margin == 0))
    assert ties / sim.n < 0.01, f"unresolved ties {ties}"


def test_kelly_returns_zero_without_an_edge():
    assert kelly_fraction(0.50, american_to_decimal(-110)) == 0.0
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_kelly_scales_with_edge():
    small = kelly_fraction(0.53, 2.0)
    large = kelly_fraction(0.60, 2.0)
    assert 0 < small < large
    # Quarter Kelly on a 60% shot at even money: 0.25 * 0.20 = 0.05
    approx(large, 0.05, tol=1e-9)


def test_one_run_game_rate_bias_is_pinned():
    """Pin the known miss on the margin distribution.

    The engine produces more one-run games than MLB actually plays, which
    drags every favourite's -1.5 down. The bias is documented in `rundist`;
    this test fails if it silently drifts, in either direction, because the
    run-line edge threshold is set from its size.
    """
    sim = simulate_game(4.5, 4.5, n_sims=250_000, seed=7)
    one_run = float(np.mean(np.abs(sim.margin) == 1))
    # Real MLB sits near 0.285. Anything outside this window means the
    # documented ~2.8pp favourite bias no longer describes the model.
    assert 0.300 < one_run < 0.322, one_run


def test_run_line_stays_inside_the_documented_uncertainty_band():
    """P(home -1.5) at pick-em across the plausible parameter range."""
    values = [
        simulate_game(4.5, 4.5, n_sims=120_000, seed=7,
                      dispersion=d).prob_cover(-1.5, home=True)
        for d in (0.35, 0.43, 0.55)
    ]
    assert max(values) - min(values) > 0.01, "band collapsed unexpectedly"
    assert max(values) - min(values) < 0.05, "band wider than documented"


def test_implied_solver_recovers_known_expected_runs():
    """Round-trip guard on the derivative pricer.

    Known expected runs are turned into a moneyline and a total, then solved
    back. Catches sign errors in the bisection, which otherwise fail silently
    by parking on a bracket rail and still printing a plausible table.
    """
    from mlbline.implied import solve

    lam_home, lam_away, line = 4.9, 4.1, 8.5
    truth = simulate_game(lam_home, lam_away, n_sims=200_000, seed=99)

    got = solve("roundtrip",
                market_home_win=truth.prob_home_win(),
                market_over=truth.prob_over(line),
                total_line=line, n_sims=40_000, price_sims=120_000)

    assert got.fit_error < 0.01, f"residual {got.fit_error}"
    assert abs(got.lam_home - lam_home) < 0.25, got.lam_home
    assert abs(got.lam_away - lam_away) < 0.25, got.lam_away
    # The point of the exercise: the derived run line must match the truth.
    assert abs(got.p_home_cover - truth.prob_cover(-1.5, home=True)) < 0.02


def test_implied_solver_reports_a_bad_fit_rather_than_hiding_it():
    """An unreachable target must surface as residual, not a silent answer."""
    from mlbline.implied import solve

    got = solve("impossible", market_home_win=0.99, market_over=0.50,
                total_line=8.5, n_sims=20_000, price_sims=40_000)
    assert got.fit_error > 0.01


def test_gate_refuses_when_handicapping_inputs_are_missing():
    """The exact situation this environment is in: prices but no baseball."""
    result = evaluate(
        policy=BettingPolicy(),
        completeness=DataCompleteness(market_prices=True),
        model_probability=None,
        market_probability=0.52,
        decimal_odds=2.10,
        n_books=16,
    )
    assert not result
    assert any("insufficient data" in r for r in result.reasons)
    assert any("no model probability" in r for r in result.reasons)
    assert result.stake_fraction == 0.0


def test_gate_refuses_before_lineups_are_posted():
    data = _full_data()
    data.confirmed_lineups = False
    result = evaluate(
        policy=BettingPolicy(), completeness=data,
        model_probability=0.60, market_probability=0.52,
        decimal_odds=2.10, n_books=16,
    )
    assert not result
    assert any("lineups" in r for r in result.reasons)


def test_gate_refuses_a_thin_market():
    result = evaluate(
        policy=BettingPolicy(), completeness=_full_data(),
        model_probability=0.60, market_probability=0.52,
        decimal_odds=2.10, n_books=2,
    )
    assert not result
    assert any("thin market" in r for r in result.reasons)


def test_gate_refuses_small_divergence_even_with_positive_ev():
    """A 1pp disagreement is inside de-vig noise regardless of the EV."""
    result = evaluate(
        policy=BettingPolicy(), completeness=_full_data(),
        model_probability=0.53, market_probability=0.52,
        decimal_odds=2.10, n_books=16,
    )
    assert not result
    assert any("divergence" in r for r in result.reasons)


def test_gate_refuses_insufficient_expected_value():
    result = evaluate(
        policy=BettingPolicy(), completeness=_full_data(),
        model_probability=0.55, market_probability=0.50,
        decimal_odds=1.87, n_books=16,
    )
    assert not result
    assert any("expected value" in r for r in result.reasons)


def test_gate_passes_and_caps_stake_at_one_percent():
    result = evaluate(
        policy=BettingPolicy(), completeness=_full_data(),
        model_probability=0.60, market_probability=0.52,
        decimal_odds=2.10, n_books=16,
    )
    assert result, result.reasons
    # Quarter Kelly here wants ~4.5% of bankroll; the cap must bind.
    approx(result.stake_fraction, 0.01)


def test_daily_exposure_is_scaled_back_to_the_cap():
    stakes = [0.01] * 8
    capped = cap_daily_exposure(stakes, BettingPolicy())
    approx(sum(capped), 0.05, tol=1e-12)
    assert all(abs(c - capped[0]) < 1e-12 for c in capped)


def test_daily_exposure_untouched_when_within_cap():
    stakes = [0.01, 0.01]
    assert cap_daily_exposure(stakes, BettingPolicy()) == stakes


if __name__ == "__main__":
    failures = []
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
