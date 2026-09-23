"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen
belegt (Nicht-Monotonie: über 50 feste Instanzen), mit denselben Auswertungsfunktionen wie die App selbst
(`ev.run_config`/`ev.sweep`/`ev.monotonicity`) - NIE über ein Ad-hoc-Skript. Positive UND negative Aussagen: im Mittel
sinkt die Lücke mit der Breite (positiv) - aber je Instanz nicht garantiert, ein schmaler Strahl scheitert, und zuverlässig
optimal ist der Strahl erst dort, wo er MEHR Knoten expandiert als A* (negativ, die zentralen ehrlichen Befunde).
Werte sind MEDIANE über gelöste Läufe; Scheiter-Quote und Optimal-Anteil beziehen sich auf ALLE Läufe."""

from dataclasses import replace
from functools import lru_cache

import pytest

import beam_algorithm as A
import beam_constants as C
import beam_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


@lru_cache(maxsize=None)
def _mono(items):
    m = ev.monotonicity(ev.Settings(), **dict(items))
    return m


def mono(**kw):
    return _mono(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall (Breite 3, Rangfolge h) --------------------------------------------------------------------------------------------------------


def test_default_numbers():
    row = cfg()
    assert row["n_solved"] == 4 and row["failed_share"] == 20.0 and row["optimal_share"] == 20.0
    near(row["gap"], 2.79, 0.3)
    near(row["gbfs_gap"], 14.71, 0.5)
    near(row["expansion_ratio"], 0.64, 0.06)
    near(row["stored_ratio"], 0.71, 0.06)
    near(row["astar_expansions"], 93, 3)
    near(row["ucs_expansions"], 121, 3)
    near(row["gbfs_expansions"], 23, 2)
    near(row["peak_width"], 6, 0.6)


def test_greedy_best_first_never_fails_but_beam_can():
    for seed in C.SWEEP_SEEDS:
        assert ev.analyse(ev.Settings(seed=seed)).gbfs.path
    assert cfg(width=1)["failed_share"] > 0


# --- Breite: im Mittel besser ------------------------------------------------------------------------------------------------------------------


def test_width_sweep_gap_falls_and_optimal_share_rises():
    for k, gap in ((1, 16.2), (2, 7.3), (3, 2.8)):
        near(cfg(width=k)["gap"], gap, 0.8)
    near(cfg(width=4)["gap"], 0.35, 0.4)
    for k in (5, 6, 8):
        near(cfg(width=k)["gap"], 0.0, 1e-9)
    for k, share in ((1, 0), (2, 0), (3, 20), (4, 40), (5, 60), (6, 60), (8, 100)):
        assert cfg(width=k)["optimal_share"] == share


def test_narrow_beams_fail_and_from_width_four_none_do():
    assert [cfg(width=k)["failed_share"] for k in (1, 2, 3)] == [20.0, 20.0, 20.0]
    assert all(cfg(width=k)["failed_share"] == 0.0 for k in (4, 5, 6, 8, 12, 16, 24))


def test_all_five_instances_are_optimal_from_width_eight():
    assert all(cfg(width=k)["optimal_share"] == 100.0 for k in (8, 12, 16, 24))


# --- Effizienz gegen A*: nur solange man Lücke oder Scheitern in Kauf nimmt --------------------------------------------------------------------------


def test_expansion_ratio_against_a_star_crosses_one_at_width_six():
    for k, ratio in ((1, 0.27), (2, 0.47), (3, 0.64), (4, 0.78), (5, 0.97), (6, 1.08), (8, 1.24)):
        near(cfg(width=k)["expansion_ratio"], ratio, 0.06)
    assert all(cfg(width=k)["expansion_ratio"] < 1.02 for k in (1, 2, 3, 4, 5))
    assert all(cfg(width=k)["expansion_ratio"] > 1.0 for k in (6, 8, 12, 16, 24))


def test_stored_ratio_by_width():
    for k, ratio in ((1, 0.45), (2, 0.60), (3, 0.71), (4, 0.85), (5, 0.94), (6, 1.00), (8, 1.10)):
        near(cfg(width=k)["stored_ratio"], ratio, 0.06)


def test_reliably_optimal_beam_is_more_expensive_than_a_star_and_saturates():
    row = cfg(width=8)
    assert row["optimal_share"] == 100.0 and row["expansion_ratio"] > 1.0
    for k in (12, 24):
        near(cfg(width=k)["expansion_ratio"], 1.25, 0.05)
        near(cfg(width=k)["beam_expansions"], 120, 3)


# --- Rangfolge f statt h -------------------------------------------------------------------------------------------------------------------------


def test_rank_f_has_smaller_gaps_at_width_two_to_four_but_more_failures_at_width_one():
    near(cfg(width=2, rank="f")["gap"], 4.5, 0.8)
    near(cfg(width=3, rank="f")["gap"], 0.35, 0.4)
    near(cfg(width=4, rank="f")["gap"], 0.0, 0.3)
    assert cfg(width=2, rank="f")["gap"] < cfg(width=2)["gap"] and cfg(width=3, rank="f")["gap"] < cfg(width=3)["gap"]
    assert cfg(width=2, rank="f")["failed_share"] == 0.0 and cfg(width=2)["failed_share"] == 20.0
    assert cfg(width=1, rank="f")["failed_share"] == 40.0 and cfg(width=1)["failed_share"] == 20.0


# --- Hindernisdichte und Rastergröße -------------------------------------------------------------------------------------------------------------


def test_obstacle_sweep_at_width_three():
    rows = ev.sweep("obstacle_pct", replace(ev.Settings(), width=3))
    for row, gap in zip(rows, (2.7, 0.0, 1.3, 0.0, 0.0)):
        near(row["gap"], gap, 0.4)
    assert [r["failed_share"] for r in rows] == [0.0, 0.0, 20.0, 0.0, 0.0]
    for row, ratio in zip(rows, (0.61, 0.72, 0.82, 0.98, 1.00)):
        near(row["expansion_ratio"], ratio, 0.06)
    assert rows[0]["expansion_ratio"] < rows[2]["expansion_ratio"] < rows[4]["expansion_ratio"] + 1e-9


def test_dense_obstacles_make_a_narrow_beam_fail():
    assert [cfg(obstacle_pct=40, width=k)["failed_share"] for k in (1, 2, 3, 4)] == [60.0, 40.0, 0.0, 0.0]


def test_size_sweep_at_width_three_a_fixed_width_stops_being_enough():
    rows = ev.sweep("side", replace(ev.Settings(), width=3))
    for row, gap in zip(rows, (0.0, 0.0, 0.0, 1.0, 5.8)):
        near(row["gap"], gap, 0.4)
    assert [r["optimal_share"] for r in rows] == [80.0, 80.0, 60.0, 40.0, 0.0]
    for row, ratio in zip(rows, (1.14, 0.84, 0.68, 0.54, 0.40)):
        near(row["expansion_ratio"], ratio, 0.06)


# --- Nicht-Monotonie -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side,obstacle,total,fails,costlier", [(12, 15, 6, 0, 6), (12, 0, 2, 0, 2), (12, 40, 10, 10, 0), (20, 15, 14, 2, 12)])
def test_non_monotone_share_by_setting_rank_h(side, obstacle, total, fails, costlier):
    m = mono(side=side, obstacle_pct=obstacle, rank="h")
    assert m["n"] == 50
    near(m["non_monotone_share"], total, 4)
    near(m["wider_fails_share"], fails, 4)
    near(m["wider_costlier_share"], costlier, 4)


def test_non_monotonicity_exists_for_rank_f_too():
    assert mono(side=12, obstacle_pct=15, rank="f")["non_monotone_share"] > 0
    assert mono(side=20, obstacle_pct=15, rank="f")["non_monotone_share"] > 0


def test_the_wider_fails_example_seed_shows_the_pattern_from_the_help_text():
    costs = ev.cost_by_width(ev.Settings(side=12, obstacle_pct=40, seed=200007))
    assert costs[0] != float("inf") and costs[1] == float("inf") and all(c != float("inf") for c in costs[2:6])


def test_greedy_best_first_solves_every_instance_where_a_beam_fails():
    for seed in C.MONO_SEEDS:
        a = ev.analyse(ev.Settings(side=12, obstacle_pct=40, seed=seed, width=1))
        assert a.gbfs.path


# --- Grenze der Schicht-Suche und Falle ----------------------------------------------------------------------------------------------------------


def test_unlimited_width_was_optimal_on_every_tested_instance():
    for obstacle in (0, 15, 40):
        for seed in range(300000, 300040):
            a = ev.analyse(ev.Settings(obstacle_pct=obstacle, seed=seed, width=10 ** 6))
            assert a.solved and a.optimal


def test_trap_numbers():
    k1 = ev.analyse(ev.Settings(network="trap", width=1))
    k2 = ev.analyse(ev.Settings(network="trap", width=2))
    assert k1.solved and 7.4 < k1.gap < 7.5 and k2.optimal
    assert (k1.beam.expansions, k1.astar.expansions) == (5, 7)
    near(k1.expansion_ratio, 0.71, 0.01)
