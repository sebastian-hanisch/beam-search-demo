import math

import beam_constants as C
import beam_evaluation as EV


def test_analyse_grid_and_trap_agree_with_the_reference_searches():
    for network in ("grid", "trap"):
        a = EV.analyse(EV.Settings(network=network, side=8, obstacle_pct=20, seed=1, width=24))
        assert a.solved and a.beam.path[0] == a.inst.start and a.beam.path[-1] == a.inst.goal
        assert abs(a.astar.cost - a.ucs.cost) < 1e-9 and a.gbfs.cost >= a.ucs.cost - 1e-9


def test_a_failed_run_is_flagged_and_has_no_gap_or_ratios():
    a = EV.analyse(EV.Settings(side=12, obstacle_pct=15, seed=100000, width=1))
    assert not a.solved and not a.optimal
    assert math.isnan(a.gap) and math.isnan(a.expansion_ratio) and math.isnan(a.stored_ratio)
    assert a.gbfs.path                                                            # Greedy Best-First hat Backtracking und findet einen Pfad


def test_gap_is_never_negative_and_optimal_matches_a_zero_gap():
    for seed in range(20):
        a = EV.analyse(EV.Settings(side=8, obstacle_pct=15, seed=seed, width=3))
        if a.solved:
            assert a.gap >= -1e-9 and a.optimal == (a.gap <= 1e-7)


def test_run_config_reports_failure_and_optimal_shares_over_all_runs_and_medians_with_ranges():
    out = EV.run_config(EV.Settings(), width=1)
    assert out["n_runs"] == 5 and out["n_solved"] == 4 and out["failed_share"] == 20.0 and out["optimal_share"] == 0.0
    assert out["gap_lo"] <= out["gap"] <= out["gap_hi"]
    wide = EV.run_config(EV.Settings(), width=24)
    assert wide["failed_share"] == 0.0 and wide["optimal_share"] == 100.0 and wide["n_solved"] == 5


def test_sweep_returns_one_row_per_value():
    assert [r["value"] for r in EV.sweep("width", EV.Settings(side=8))] == list(C.WIDTHS)
    assert [r["value"] for r in EV.sweep("obstacle_pct", EV.Settings(side=8))] == list(C.OBSTACLE_SWEEP)
    assert [r["value"] for r in EV.sweep("side", EV.Settings(), values=(6, 8))] == [6, 8]


def test_analyse_is_deterministic():
    a1 = EV.analyse(EV.Settings(side=9, obstacle_pct=25, seed=7))
    a2 = EV.analyse(EV.Settings(side=9, obstacle_pct=25, seed=7))
    assert a1.beam.path == a2.beam.path and a1.beam.expansions == a2.beam.expansions


def test_is_non_monotone_classifies_the_three_cases():
    inf = float("inf")
    assert EV.is_non_monotone([5.0, 4.0, 3.0, 3.0]) == (False, False, False)      # monoton
    assert EV.is_non_monotone([inf, 5.0, inf, 4.0]) == (True, True, False)         # breiter scheitert trotz Erfolg
    assert EV.is_non_monotone([5.0, 4.0, 4.5, 4.0]) == (True, False, True)         # breiter kostet mehr
    assert EV.is_non_monotone([inf, inf, 4.0]) == (False, False, False)            # Scheitern -> Erfolg ist keine Verschlechterung


def test_monotonicity_counts_match_cost_by_width():
    s = EV.Settings(side=8, obstacle_pct=15)
    m = EV.monotonicity(s, seeds=(1, 2, 3, 4), widths=(1, 2, 3, 4))
    assert m["n"] == 4 and len(m["rows"]) == 4
    for seed, costs, worse, fail, cost in m["rows"]:
        assert costs == EV.cost_by_width(EV.Settings(side=8, obstacle_pct=15, seed=seed), widths=(1, 2, 3, 4))
        assert (worse, fail, cost) == EV.is_non_monotone(costs)
    assert m["non_monotone_share"] == 100.0 * sum(r[2] for r in m["rows"]) / 4
