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


def test_slate_selection_refuses_a_repeated_pairing():
    """Series mean the same pairing appears on consecutive days.

    An odds feed pulled late in the evening carries tomorrow's lines beside
    tonight's. Indexing by matchup alone let the later game overwrite the
    earlier one, which priced tomorrow's pitchers against tonight's board and
    produced a +32% handicap. Better to refuse than to guess.
    """
    from datetime import datetime, timezone

    from mlbline.odds import Game, eastern_date, select_slate

    tonight = Game("a", datetime(2026, 8, 15, 2, 11, tzinfo=timezone.utc),
                   "Los Angeles Dodgers", "Milwaukee Brewers")
    tomorrow = Game("b", datetime(2026, 8, 15, 23, 15, tzinfo=timezone.utc),
                    "Los Angeles Dodgers", "Milwaukee Brewers")

    # A 10pm Eastern first pitch is already tomorrow in UTC.
    assert eastern_date(tonight.commence_time) == "2026-08-14"
    assert eastern_date(tomorrow.commence_time) == "2026-08-15"

    try:
        select_slate([tonight, tomorrow])
    except ValueError as exc:
        assert "Pass a date" in str(exc)
    else:
        raise AssertionError("duplicate pairing was silently resolved")

    chosen = select_slate([tonight, tomorrow], "2026-08-14")
    assert len(chosen) == 1
    assert chosen[tonight.matchup].game_id == "a"


def test_a_same_day_doubleheader_is_numbered_not_refused():
    """Two dates is a series; two on one date is a doubleheader.

    The first must raise, because pairing tomorrow's lines with tonight's
    board is a real failure. The second must resolve, because both games are
    genuinely on the slate and each needs its own price.
    """
    from datetime import datetime, timezone

    from mlbline.odds import Game, select_slate

    early = Game("g1", datetime(2026, 8, 17, 17, 41, tzinfo=timezone.utc),
                 "Cincinnati Reds", "St. Louis Cardinals")
    late = Game("g2", datetime(2026, 8, 17, 22, 41, tzinfo=timezone.utc),
                "Cincinnati Reds", "St. Louis Cardinals")

    chosen = select_slate([late, early], "2026-08-17")
    assert set(chosen) == {f"{early.matchup} G1", f"{early.matchup} G2"}
    # Numbered by start time, not by feed order.
    assert chosen[f"{early.matchup} G1"].game_id == "g1"
    assert chosen[f"{early.matchup} G2"].game_id == "g2"
    # The bare matchup must not resolve to an arbitrary half.
    assert early.matchup not in chosen


def test_bias_correction_does_not_touch_moneyline_equivalent_handicaps():
    """A handicap under one run is a moneyline bet, and carries no error.

    The engine's win probability is fitted to the market moneyline, so laying
    a handicap of zero — or of 0.5, which still only asks whether the
    favourite won — has nothing for a margin-distribution correction to fix.
    Applying one anyway manufactured +2.5% of edge from nothing, and put a
    pick-em third on a slate.
    """
    from mlbline.asian import AsianLine, handicap_ev

    rng = np.random.default_rng(3)
    margin = rng.integers(-6, 7, 40_000)

    for line in (AsianLine(0.0, 0.0, 0.0, "0"),
                 AsianLine(0.5, 0.5, 0.0, "0.5")):
        for laying in (True, False):
            plain = handicap_ev(margin, line, hk_price=0.95, laying=laying)
            shifted = handicap_ev(margin, line, hk_price=0.95,
                                  laying=laying, bias=0.0128)
            approx(plain, shifted, tol=1e-12)


def test_bias_correction_applies_where_the_margin_actually_matters():
    """At a 1.5 line the correction moves both sides by the full amount."""
    from mlbline.asian import AsianLine, handicap_ev

    rng = np.random.default_rng(3)
    margin = rng.integers(-6, 7, 40_000)
    line = AsianLine(1.5, 1.5, 0.0, "1.5")
    delta = 0.0128

    lay = handicap_ev(margin, line, hk_price=0.95, laying=True)
    lay_adj = handicap_ev(margin, line, hk_price=0.95, laying=True, bias=delta)
    approx(lay_adj - lay, delta * 1.95, tol=1e-12)

    get = handicap_ev(margin, line, hk_price=0.95, laying=False)
    get_adj = handicap_ev(margin, line, hk_price=0.95, laying=False, bias=delta)
    approx(get_adj - get, -delta * 1.95, tol=1e-12)


def test_bias_correction_is_asymmetric_at_an_integer_line():
    """A push returns stake, so the two sides do not move together.

    At a line of exactly one the misallocated probability sits between
    "wins by two" and the push, not the loss, so laying gains delta*price
    while receiving loses only delta.
    """
    from mlbline.asian import AsianLine, handicap_ev

    rng = np.random.default_rng(3)
    margin = rng.integers(-6, 7, 40_000)
    line = AsianLine(1.0, 1.0, 0.0, "1")
    delta = 0.0128

    lay_gap = (handicap_ev(margin, line, hk_price=0.95, laying=True, bias=delta)
               - handicap_ev(margin, line, hk_price=0.95, laying=True))
    get_gap = (handicap_ev(margin, line, hk_price=0.95, laying=False, bias=delta)
               - handicap_ev(margin, line, hk_price=0.95, laying=False))
    approx(lay_gap, delta * 0.95, tol=1e-12)
    approx(get_gap, -delta, tol=1e-12)
    assert abs(lay_gap) != abs(get_gap)


def test_asian_line_sign_convention():
    """A minus reaches up toward the half above, a plus reaches down.

    Getting this backwards puts a line most of a run from the market and
    manufactures huge fake edges -- a total read as 8.45 when the board said
    7.55 priced the under at +17%.
    """
    from mlbline.asian import parse_line

    approx(parse_line("9-50").effective, 9.25)
    approx(parse_line("9+30").effective, 8.85)
    approx(parse_line("8+90").effective, 7.55)
    approx(parse_line("8平").effective, 8.0)
    approx(parse_line("9.5").effective, 9.5)
    approx(parse_line("0").effective, 0.0)
    # A one-digit weight is hundredths, not tenths.
    approx(parse_line("1+5").effective, 0.975)


def test_fit_does_not_move_when_only_the_modal_line_changes():
    """Books migrating between lines must not move the fitted total.

    The Rays' totals read 0.5217 at 7.0 in two snapshots ten hours apart --
    an unchanged market -- but books drifted from 7.5 to 7.0 and flipped
    which line was modal. Anchoring on the modal line alone shifted the
    fitted total 0.176 runs and moved a candidate's EV by 3.6 points on
    nothing at all. Fitting every quoted line, weighted by book count, has
    to be far more stable than that.
    """
    from mlbline.implied import solve

    same = dict(market_home_win=0.602, n_sims=20_000, price_sims=20_000)
    # One market, described from either line's point of view.
    high = solve("t", market_over=0.4767, total_line=7.5, **same)
    low = solve("t", market_over=0.5217, total_line=7.0, **same)
    single_gap = abs(high.mean_total - low.mean_total)

    anchors = [(7.0, 0.5217, 11.0), (7.5, 0.4767, 4.0)]
    both_high = solve("t", market_over=0.4767, total_line=7.5,
                      over_anchors=anchors, **same)
    both_low = solve("t", market_over=0.5217, total_line=7.0,
                     over_anchors=anchors, **same)

    # Identical anchors and identical weights must give an identical fit,
    # whatever the caller happens to name as the modal line.
    approx(both_high.mean_total, both_low.mean_total, tol=1e-9)
    assert single_gap > 0.1, f"expected the single-anchor fit to drift, got {single_gap}"

    # And the residual has to actually report disagreement between lines
    # rather than the zero a single self-consistent anchor always gives.
    assert both_high.anchor_spread > 0.0
    assert solve("t", market_over=0.5217, total_line=7.0,
                 **same).anchor_spread < 0.01


def test_pickem_and_half_run_handicaps_are_the_same_bet():
    """Laying 0 and laying 0.5 are one wager, so they cannot differ.

    Baseball has no ties: winning by at least half a run and winning
    outright are the same event. The prices already agree, but the decode
    report differenced the literal numbers and charged a board hanging a
    flat 0 with a 0.5-run error against a fair line of 0.5. One pick'em on
    a nine-game slate was enough to double the reported handicap error,
    which is the number used to decide whether a board was read correctly.
    """
    from mlbline.asian import parse_line, handicap_ev

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from price_board import comparable_handicap

    rng = np.random.default_rng(11)
    margin = rng.integers(-8, 9, 50_000)
    margin = margin[margin != 0]

    for laying in (True, False):
        approx(handicap_ev(margin, parse_line("0"), hk_price=0.95,
                           laying=laying),
               handicap_ev(margin, parse_line("0.5"), hk_price=0.95,
                           laying=laying),
               tol=1e-12)

    # And the decode check must see them as the same line, not half a run
    # apart, while leaving every real handicap alone.
    approx(comparable_handicap(0.0), comparable_handicap(0.5))
    approx(comparable_handicap(1.325), 1.325)
    approx(comparable_handicap(0.675), 0.675)


def test_integer_lines_push_and_half_lines_do_not():
    from mlbline.asian import AsianLine, total_ev

    runs = np.array([7, 8, 8, 9, 10])
    whole = AsianLine(8.0, 8.0, 0.0, "8平")
    half = AsianLine(8.5, 8.5, 0.0, "8.5")

    # At 8 the two eights push, so only 9 and 10 win and only 7 loses.
    approx(total_ev(runs, whole, hk_price=1.0, over=True), (2 - 1) / 5)
    # At 8.5 nothing pushes: 9 and 10 win, 7 and both 8s lose.
    approx(total_ev(runs, half, hk_price=1.0, over=True), (2 - 3) / 5)


def test_asian_split_ticket_interpolates_between_its_legs():
    from mlbline.asian import AsianLine, total_ev

    runs = np.array([7, 8, 8, 9, 10])
    low = total_ev(runs, AsianLine(8.0, 8.0, 0.0, ""), hk_price=1.0, over=True)
    high = total_ev(runs, AsianLine(8.5, 8.5, 0.0, ""), hk_price=1.0, over=True)
    mixed = total_ev(runs, AsianLine(8.0, 8.5, 0.4, ""), hk_price=1.0, over=True)
    approx(mixed, 0.6 * low + 0.4 * high, tol=1e-9)


def test_fair_total_is_not_pinned_to_an_integer():
    """Regression on a solver that could only ever return whole numbers.

    Bisecting on the line itself cannot work, because `runs > line` only
    changes at integers. The earlier version returned 8.00 or 9.00 for every
    game and so could not detect a decoding error at all.
    """
    from mlbline.asian import fair_total

    rng = np.random.default_rng(1)
    runs = rng.poisson(8.5, 120_000)
    fair = fair_total(runs)
    assert abs(fair - round(fair)) > 0.05, f"snapped to integer: {fair}"
    # Over 8.5 wins under half the time here, so a fair line sits below it.
    assert 7.8 < fair < 8.5, fair


def test_fair_total_tracks_the_scoring_level():
    from mlbline.asian import fair_total

    rng = np.random.default_rng(2)
    low = fair_total(rng.poisson(7.0, 80_000))
    high = fair_total(rng.poisson(10.0, 80_000))
    assert high > low + 2.0, (low, high)


def test_real_game_history_is_intact():
    """Guard the Retrosheet extract the validation is measured against."""
    import csv

    path = Path(__file__).resolve().parents[1] / "data" / "history" / "games.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 7289, len(rows)

    home = np.array([int(r["home_runs"]) for r in rows])
    away = np.array([int(r["away_runs"]) for r in rows])
    margin = home - away

    # Three complete seasons of real baseball. If these move, the file
    # changed, and every calibration claim built on it needs rechecking.
    approx(float(np.mean(np.abs(margin) == 1)), 0.2832, tol=0.002)
    approx(float(np.mean(margin > 0)), 0.5285, tol=0.002)
    approx(float(np.mean(margin >= 2) / np.mean(margin > 0)), 0.6734, tol=0.003)
    approx(float(np.mean(margin <= -2) / np.mean(margin < 0)), 0.7655, tol=0.003)


def test_markov_reproduces_the_share_of_wins_by_two_or_more():
    """The quantity that sets run-line prices, checked against real games.

    A home team's share of wins that clear two runs is what a -1.5 line is,
    arithmetically. Real 2023-2025 baseball puts it at 67.3%. The old engine
    said 64.6%, and that 2.7pp shortfall is almost exactly the systematic
    disagreement it showed against the market -- the bias was the model, not
    the market.
    """
    from mlbline.markov import simulate_game as markov_game

    rng = np.random.default_rng(4)
    lam_home = np.clip(rng.normal(4.59, 1.30, 40), 2.2, 8.0).round(1)
    lam_away = np.clip(rng.normal(4.37, 1.30, 40), 2.2, 8.0).round(1)

    margins = np.concatenate([
        markov_game(float(h), float(a), n_sims=4000, seed=int(i)).margin
        for i, (h, a) in enumerate(zip(lam_home, lam_away))
    ])
    share = float(np.mean(margins >= 2) / np.mean(margins > 0))
    assert abs(share - 0.6734) < 0.02, share


def test_markov_half_inning_distribution_matches_league():
    """The chain must reproduce how often a half-inning scores 0, 1, 2 ..."""
    from mlbline.markov import _run_pmf, scale_rates

    pmf = _run_pmf(scale_rates(4.5), seed=7)
    for runs, actual in ((0, 0.727), (1, 0.150), (2, 0.064), (3, 0.031)):
        assert abs(pmf[runs] - actual) < 0.015, (runs, pmf[runs])


def test_markov_scale_rates_hits_its_target():
    from mlbline.markov import runs_per_nine, scale_rates

    for target in (3.5, 4.5, 5.5):
        assert abs(runs_per_nine(scale_rates(target)) - target) < 0.1


def test_walkoff_margins_are_not_all_one_run():
    """The reason the chain exists.

    The old engine truncated a home team's winning rally to a one-run margin
    and bolted on a fixed overshoot rate. Here play simply stops on the
    go-ahead run, so a walk-off single scores one and a grand slam scores
    four, in whatever mix the batting events produce.
    """
    from mlbline.markov import simulate_game as markov_game

    sim = markov_game(4.5, 4.5, n_sims=150_000, seed=21)
    # Home wins are the only ones that can end mid-inning.
    home_wins = sim.margin[sim.margin > 0]
    by_one = float(np.mean(home_wins == 1))
    assert 0.30 < by_one < 0.45, by_one
    # Multi-run walk-offs must appear without being hard-coded.
    assert float(np.mean(home_wins >= 3)) > 0.20


def test_markov_beats_old_engine_on_margin_distribution():
    """Head to head against the league margin distribution."""
    from mlbline.markov import simulate_game as markov_game

    real = {1: 0.285, 2: 0.190, 3: 0.150, 4: 0.115, 5: 0.080}
    rng = np.random.default_rng(5)
    matchups = np.clip(rng.normal(4.5, 0.75, size=(60, 2)), 2.6, 7.0).round(1)

    def league_error(fn):
        margins = np.concatenate([
            np.abs(fn(float(h), float(a), n_sims=2000,
                      seed=int(h * 100 + a * 7)).margin)
            for h, a in matchups
        ])
        return sum(abs(float(np.mean(margins == k)) - v)
                   for k, v in real.items())

    assert league_error(markov_game) < league_error(simulate_game)


def test_markov_leaves_no_unresolved_ties():
    from mlbline.markov import simulate_game as markov_game

    sim = markov_game(4.5, 4.5, n_sims=60_000, seed=23)
    assert float(np.mean(sim.margin == 0)) < 0.001


def test_markov_run_line_stays_consistent_with_moneyline():
    from mlbline.markov import simulate_game as markov_game

    sim = markov_game(5.2, 4.0, n_sims=120_000, seed=25)
    assert sim.prob_cover(-1.5, home=True) < sim.prob_home_win()
    approx(sim.prob_cover(-1.5, home=True) + sim.prob_cover(1.5, home=False),
           1.0, tol=1e-6)


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
