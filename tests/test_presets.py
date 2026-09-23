"""Presets: Vollständigkeit, gültige Werte, der Median des Expansions-Verhältnisses bleibt in der gemessenen Spannweite
über die 5 festen Sweep-Instanzen (vollständig deterministisch), und jedes Preset zeigt tatsächlich, was sein Name sagt."""

import pytest

import beam_algorithm as A
import beam_constants as C
import beam_evaluation as ev
import beam_presets as P


def _settings(p, seed=None):
    return ev.Settings(network=p["network"], side=p["side"], obstacle_pct=p["obstacle_pct"], seed=p["seed"] if seed is None else seed,
                       width=p["width"], rank=p["rank"])


def test_every_preset_has_help_and_a_band():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 7
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid():
    for p in C.PRESETS.values():
        assert p["network"] in P.NETWORKS and p["width"] in C.WIDTHS and p["rank"] in C.RANKS
        assert C.SIDE_MIN <= p["side"] <= C.SIDE_MAX and C.OBSTACLE_MIN <= p["obstacle_pct"] <= C.OBSTACLE_MAX


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_expansion_ratio_stays_in_its_measured_band(name):
    p = C.PRESETS[name]
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    base = _settings(p)
    row = ev.run_config(base) if p["network"] == "grid" else ev.run_config(base, seeds=(p["seed"],))
    assert lo <= row["expansion_ratio"] <= hi, row["expansion_ratio"]


def test_trap_preset_falls_for_the_bait_like_greedy_best_first_and_width_two_recovers():
    a = ev.analyse(_settings(C.PRESETS["Handgebaute Falle (Breite 1)"]))
    assert a.solved and abs(a.gap - a.gbfs_gap) < 1e-9 and 7.0 < a.gap < 8.0
    wider = ev.analyse(ev.Settings(network="trap", width=2))
    assert wider.optimal


def test_lost_goal_preset_fails_although_greedy_best_first_finds_a_path():
    a = ev.analyse(_settings(C.PRESETS["Ziel verloren (Breite 1)"]))
    assert not a.solved and a.beam.path == [] and a.gbfs.path


def test_wider_fails_preset_narrow_succeeds_width_two_fails_width_three_succeeds():
    p = C.PRESETS["Breiter scheitert (Breite 2)"]
    by_width = {k: ev.analyse(ev.Settings(side=p["side"], obstacle_pct=p["obstacle_pct"], seed=p["seed"], width=k, rank=p["rank"])).solved for k in (1, 2, 3)}
    assert by_width == {1: True, 2: False, 3: True}


def test_wider_costs_more_preset_width_three_is_worse_than_width_two():
    p = C.PRESETS["Breiter ist teurer (Breite 3)"]
    costs = {k: ev.analyse(ev.Settings(side=p["side"], obstacle_pct=p["obstacle_pct"], seed=p["seed"], width=k, rank=p["rank"])).beam.cost for k in (2, 3)}
    assert costs[3] > costs[2] + 1.0
    assert costs[2] == pytest.approx(190.3, abs=0.1) and costs[3] == pytest.approx(193.6, abs=0.1)     # Zahlen im Hilfetext


def test_width_eight_preset_expands_more_than_a_star_on_its_instance():
    a = ev.analyse(_settings(C.PRESETS["Breite 8 (teurer als A*)"]))
    assert a.optimal and a.beam.expansions > a.astar.expansions
    assert (a.beam.expansions, a.astar.expansions) == (118, 82)                                       # Zahlen im Hilfetext


def test_bounds_and_permalink_constants():
    assert P.bounds("side_slider") == (C.SIDE_MIN, C.SIDE_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_WIDTH in C.WIDTHS and C.DEFAULT_RANK in C.RANKS


def test_network_width_and_rank_permalink_casters():
    assert P._network_from_str("trap") == "trap"
    with pytest.raises(ValueError):
        P._network_from_str("nope")
    assert P.SETTING_SPECS["width_select"].caster("8") == 8
    with pytest.raises(ValueError):
        P.SETTING_SPECS["width_select"].caster("7")
    assert P.SETTING_SPECS["rank_select"].caster("f") == "f"
    with pytest.raises(ValueError):
        P.SETTING_SPECS["rank_select"].caster("g")


def test_width_one_preset_is_the_greedy_descent_of_the_algorithm_module():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    inst = ev.instance(p["side"], p["obstacle_pct"], p["seed"])
    a = A.beam_search(inst.graph, inst.start, inst.goal, 1, "h")
    assert a.cost == pytest.approx(215.53, abs=0.01)                                                  # gleich der Wurzel-Lücke von 22.5 %
