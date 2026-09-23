"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jede Schicht, beide Instanz-Typen, gescheiterte Läufe,
Randwerte, Würfel-Knopf, Permalink-Grenzen, Instanzwechsel, Sweeps und Experiment auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import beam_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="beam_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_four_metrics():
    at = _run()
    _ok(at)
    assert {"Lücke zum Optimum", "Expansionen / A*", "Speicher / A*", "Status"} <= {m.label for m in at.metric}
    assert _metric(at, "Status") == "Gelöst"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["network_select"] == p["network"] and at.session_state["width_select"] == p["width"] and at.session_state["rank_select"] == p["rank"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_both_networks(step):
    for network in ("grid", "trap"):
        at = _run(network_select=network, step=step)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["beam_step"] == step


@pytest.mark.parametrize("step", [1, 2, 3])
def test_failed_runs_render_in_every_step_and_are_flagged(step):
    at = _run(seed_input=100000, width_select=1, step=step)
    _ok(at)
    assert _metric(at, "Status") == "Gescheitert" and _metric(at, "Lücke zum Optimum") == "-"
    if step == 3:
        assert any("gescheitert" in w.value for w in at.warning)


def test_layer_slider_walks_through_all_layers_and_survives_an_instance_change():
    at = _run(step=2, width_select=2)
    _ok(at)
    slider = at.slider(key="beam_layer")
    slider.set_value(slider.max).run()
    _ok(at)
    assert at.session_state["beam_layer"] == slider.max
    at.session_state["network_select"] = "trap"                  # weniger Schichten: gespeicherter Wert wird geklemmt
    at.run()
    _ok(at)
    assert at.session_state["beam_layer"] <= 3


@pytest.mark.parametrize("kw", [
    dict(side_slider=C.SIDE_MIN), dict(side_slider=C.SIDE_MAX), dict(obstacle_slider=C.OBSTACLE_MIN), dict(obstacle_slider=C.OBSTACLE_MAX),
    dict(width_select=C.WIDTHS[0]), dict(width_select=C.WIDTHS[-1], side_slider=C.SIDE_MAX), dict(rank_select="f"),
])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))
    _ok(_run(step=2, **kw))


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["side"] = "9999"
    at.query_params["obstacle"] = "9999"
    at.query_params["width"] = "7"
    at.query_params["rank"] = "zzz"
    at.run()
    _ok(at)
    assert at.session_state["side_slider"] == C.SIDE_MAX and at.session_state["obstacle_slider"] == C.OBSTACLE_MAX
    assert at.session_state["width_select"] == C.DEFAULT_WIDTH and at.session_state["rank_select"] == C.DEFAULT_RANK


def test_sidebar_hides_grid_only_controls_for_the_trap_network_but_keeps_width_and_rank():
    at = _run(network_select="trap")
    _ok(at)
    assert not any(s.key == "side_slider" for s in at.slider)
    assert any(s.key == "width_select" for s in at.select_slider) and any(r.key == "rank_select" for r in at.radio)


def test_changing_the_instance_while_on_step_two_does_not_crash():
    at = _run(step=2, side_slider=12)
    _ok(at)
    at.session_state["side_slider"] = C.SIDE_MIN
    at.run()
    _ok(at)
    at.session_state["network_select"] = "trap"
    at.run()
    _ok(at)


@pytest.mark.parametrize("param", ["width", "obstacle_pct", "side"])
@pytest.mark.parametrize("metric", ["gap", "expansion_ratio", "stored_ratio", "shares"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(side_slider=8, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_monotonicity_experiment_runs_on_demand_and_shows_three_metrics():
    at = _run(side_slider=8)
    next(b for b in at.button if b.key == "mono_start").click().run()
    _ok(at)
    labels = {m.label for m in at.metric}
    assert {"Instanzen mit schlechterem breiteren Strahl", "... davon: breiter scheitert", "... davon: breiter kostet mehr"} <= labels


def test_cost_curve_is_shown_for_the_trap_without_the_instance_experiment():
    at = _run(network_select="trap")
    _ok(at)
    assert not any(b.key == "mono_start" for b in at.button)
    assert any(c.value == "🔬 Wird es mit mehr Breite immer besser?" for c in at.subheader)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Monobeam" in m.value and "Beam Stack Search" in m.value for m in at.markdown)
