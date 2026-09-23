"""Beam Search - wie viele Kandidaten reichen, wenn es kein Zurück gibt? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Heuristische-Baumsuche-Linie der "Konzepte"-Reihe, Kind der Wurzel (Greedy Best-First): GBFS
verfolgt EINEN Kandidaten und ist kurzsichtig. Beam Search hält je Schicht die k besten Kandidaten parallel - aber
ohne Zurück: was aus dem Strahl fällt, ist verloren. Wie viel Breite braucht es, und ist mehr Breite immer besser?
Muss gemessen werden, nicht angenommen.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import beam_constants as C
from beam_evaluation import SWEEP_LABELS, Settings, analyse, cost_by_width, monotonicity, sweep
from beam_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from beam_visualization import (
    ASTAR_COLOR,
    BEAM_COLOR,
    build_cost_by_width,
    build_instance,
    build_layer_map,
    build_paths,
    build_share_bars,
    build_sweep,
)

st.set_page_config(page_title="Beam Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _costs(settings):
    return cost_by_width(settings)


@st.cache_data(show_spinner=False)
def _monotonicity(base):
    m = monotonicity(base)
    m.pop("rows")
    return m


st.title("🔦 Beam Search – wie viele Kandidaten reichen, wenn es kein Zurück gibt?")
st.markdown(
    """
**Viertes Stück der Heuristische-Baumsuche-Linie** - ein Kind der Wurzel, Greedy Best-First. Die Wurzel verfolgt
**einen** Kandidaten und ist deshalb kurzsichtig (Lücke zum Optimum, Sackgassen).

**Beam Search** hält je Schicht die **k besten** Kandidaten parallel und beschneidet den Rest - **ohne Zurück**: was aus
dem Strahl fällt, ist unwiederbringlich verloren. Damit ist die Breite k ein Regler zwischen "ein Pfad" (k = 1) und
"fast alles". Wie viel Breite braucht es? Ist mehr Breite immer besser? Und wann ist der Strahl billiger als A\\*?
"""
)
st.caption(
    "Setzt auf [greedy-best-first-demo](https://github.com/sebastian-hanisch/greedy-best-first-demo) auf (derselbe Graph, "
    "dieselbe Instanz; GBFS, A\\* und Uniform-Cost als Vergleich). Noch nicht gebaute Geschwister: Diverse Beam Search, "
    "Monobeam, Monte Carlo Tree Search (MCTS), Beam Search + A\\* → Beam Stack Search."
)

with st.expander("So funktioniert Beam Search", expanded=True):
    st.markdown(
        """
1. **Schichten:** Schicht 0 = {Start}. In jeder Schicht werden **alle** Strahl-Knoten expandiert und ihre Nachfolger
   gesammelt (je Knoten das kleinste g; bereits expandierte Knoten früherer Schichten kommen nicht wieder auf).
2. **Beschneiden:** die Kandidaten werden nach der **Rangfolge** sortiert - nur **h** (Abstand zum Ziel, wie die Wurzel)
   oder **f = g + h** (wie A\\*) - und nur die besten **k** bleiben als Strahl der nächsten Schicht.
3. **Ziel erzeugt** → Ende, mit dem günstigsten Weg zum Ziel aus dieser Schicht. **Keine Kandidaten mehr** →
   **gescheitert**: kein Pfad wird behauptet, obwohl es einen gibt.
4. **Kein Zurück:** was beim Beschneiden herausfiel, wird nie wieder betrachtet. Schmal = billig, aber der richtige
   Weg kann verloren gehen. Breit = sicherer, aber teurer - und, wie die Messung zeigt, nicht garantiert besser.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    network = st.radio("Instanz", options=["grid", "trap"], format_func=lambda n: "Raster" if n == "grid" else "Handgebaute Heuristik-Falle",
                        key="network_select", horizontal=True, help="Die Falle ist ein fester, von Hand gebauter Graph - Rastergröße/Hindernisdichte/Seed wirken dort nicht.")
    if network == "grid":
        side = st.slider("Rastergröße (Seitenlänge)", *bounds("side_slider"), key="side_slider",
                          help="Eine feste Breite reicht bei größeren Rastern nicht mehr: bei Breite 3 wächst die Lücke von 0 % (Größe 6-14) auf 5.8 % (Größe 22).")
        obstacle_pct = st.slider("Hindernisdichte [%]", *bounds("obstacle_slider"), key="obstacle_slider", step=C.OBSTACLE_STEP,
                                  help="Bei dichten Hindernissen scheitert ein schmaler Strahl häufig (40 %: Breite 1 in 3 von 5 Instanzen, Breite 2 in 2 von 5).")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        side, obstacle_pct, seed = C.DEFAULT_SIDE, C.DEFAULT_OBSTACLE, C.DEFAULT_SEED
    width = st.select_slider("Strahlbreite k", options=list(C.WIDTHS), key="width_select",
                             help="Kandidaten, die je Schicht im Strahl bleiben. Breite 1 = gieriger Abstieg ohne Zurück.")
    rank = st.radio("Rangfolge", options=list(C.RANKS), format_func=lambda r: C.RANK_LABELS[r], key="rank_select", horizontal=True,
                    help="h: nur Abstand zum Ziel (Kind der Wurzel). f = g + h: mit bisherigem Weg (wie A*). Bei Breite 2-4 kleinere Lücke, aber bei Breite 1 mehr Ausfälle.")

sync_query_params({"network_select": network, "side_slider": int(side), "obstacle_slider": int(obstacle_pct), "seed_input": int(seed),
                   "width_select": int(width), "rank_select": rank})

settings = Settings(network, int(side), int(obstacle_pct), int(seed), int(width), rank)
with st.spinner("Rechne..."):
    a = _analysis(settings)
beam = a.beam

# --- Beam Search in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Beam Search in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Strahl je Schicht", 3: "3 · Ergebnis"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="beam_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    st.markdown(f"**{a.inst.graph.n} Zellen** ({len(a.inst.blocked_xy)} Hindernisse), Start (grün) und Ziel (rot)")
    st.plotly_chart(build_instance(a.inst), width="stretch", key="s1_map")
elif step == 2:
    n_layers = len(beam.per_layer)
    if n_layers > 1:
        if "beam_layer" in st.session_state:
            st.session_state["beam_layer"] = min(max(1, int(st.session_state["beam_layer"])), n_layers)
        layer = st.slider("Schicht", 1, n_layers, key="beam_layer", help="Schicht = Kantenzahl vom Start. Die Karte zeigt, was in dieser Schicht passiert.")
    else:
        layer = 1
    beam_nodes, dropped = beam.per_layer[layer - 1]
    st.markdown(
        f"**Schicht {layer} von {n_layers}:** Strahl {len(beam_nodes)} Knoten (Breite k = {settings.width}), "
        f"**{len(dropped)} Kandidaten verworfen** - für immer verloren."
        + (" **Hier ist die Suche zu Ende: keine Kandidaten mehr, das Ziel wurde nicht erreicht.**" if beam.failed and layer == n_layers else "")
        + (" Hier wird das Ziel erzeugt." if (not beam.failed) and layer == n_layers else "")
    )
    st.plotly_chart(build_layer_map(a.inst, beam.per_layer, layer - 1), width="stretch", key=f"s2_map_{layer}")
    st.caption(
        f"Insgesamt {beam.expansions} Expansionen ({a.astar.expansions} bei A\\*, {a.gbfs.expansions} bei Greedy Best-First), "
        f"größte Kandidatenzahl einer Schicht: {beam.peak_width}."
    )
else:
    gbfs_gap = f"Greedy Best-First: {a.gbfs.cost:.2f} km ({a.gbfs_gap:.2f} % zu lang)"
    if beam.failed:
        st.warning(f"Beam Search (Breite {settings.width}, Rangfolge {C.RANK_LABELS[settings.rank]}) ist **gescheitert** und behauptet keinen Pfad - "
                   f"A\\* fand {a.astar.cost:.2f} km, {gbfs_gap}.")
        st.plotly_chart(build_paths(a.inst, [], a.gbfs.path, a.astar.path), width="stretch", key="s3_map")
    else:
        st.markdown(f"**Beam Search:** {beam.cost:.2f} km – **A\\* (optimal):** {a.astar.cost:.2f} km – Lücke **{a.gap:.2f} %** – {gbfs_gap}")
        st.plotly_chart(build_paths(a.inst, beam.path, a.gbfs.path, a.astar.path), width="stretch", key="s3_map")

st.markdown("---")

# --- Ergebnis ----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Strahl gegen A* bringt")
st.caption(
    "**Lücke:** Kosten Beam gegenüber dem Optimum (Uniform-Cost). **Expansionen / A\\*** und **Speicher / A\\*** (gespeicherte = entdeckte "
    "Knoten wie bei A\\*): unter 1 heißt, der Strahl schaut sich weniger an als A\\*."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Lücke zum Optimum", "-" if beam.failed else f"{a.gap:.2f} %", delta="kein Pfad" if beam.failed else f"Greedy: {a.gbfs_gap:.2f} %", delta_color="off")
m2.metric("Expansionen / A*", "-" if beam.failed else f"{a.expansion_ratio:.2f}x", delta=f"{beam.expansions} gegen {a.astar.expansions}", delta_color="off")
m3.metric("Speicher / A*", "-" if beam.failed else f"{a.stored_ratio:.2f}x", delta=f"{beam.stored} gegen {a.astar.stored}", delta_color="off")
m4.metric("Status", "Gescheitert" if beam.failed else "Gelöst", delta=f"{beam.layers} Schichten, Spitze {beam.peak_width}", delta_color="off")

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

if network == "grid":
    st.subheader("📐 Wie hängt das Ergebnis von Breite, Hindernissen und Größe ab?")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
    metric = st.radio("Kennzahl", options=["gap", "expansion_ratio", "stored_ratio", "shares"],
                       format_func=lambda k: {"gap": "Lücke (%)", "expansion_ratio": "Expansionen / A*", "stored_ratio": "Speicher / A*", "shares": "Optimal / gescheitert (%)"}[k],
                       key="sweep_metric", horizontal=True)
    base_sweep = replace(settings, seed=0)
    if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
        st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
    if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
            rows_sweep = _sweep(sweep_param, base_sweep)
        if metric == "shares":
            st.plotly_chart(build_share_bars(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_shares")
        else:
            label = {"gap": "Lücke zum Optimum (%)", "expansion_ratio": "Expansionen Beam / A*", "stored_ratio": "Gespeicherte Knoten Beam / A*"}[metric]
            ref = 1.0 if metric != "gap" else None
            st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], metric, label, color=BEAM_COLOR if metric != "stored_ratio" else ASTAR_COLOR,
                                        ref_line=ref, ref_label="so viel wie A*" if ref else None),
                            width="stretch", key="sweep_chart")
        st.caption(
            "Median über 5 feste Instanzen (Seeds 100000–100004), Band = Minimum bis Maximum. Lücke und Verhältnisse nur über gelöste "
            "Läufe - der Balkendiagramm-Modus zeigt daneben, wie oft der Strahl optimal war und wie oft er scheiterte."
        )

    st.markdown("---")

# --- Experiment: Nicht-Monotonie ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wird es mit mehr Breite immer besser?")
st.caption("Pfadkosten dieser einen Instanz für jede Breite von 1 bis 24 (Rangfolge wie in der Seitenleiste). Wo die Kurve steigt oder ein x auftaucht, ist ein breiterer Strahl schlechter.")
costs = _costs(settings)
st.plotly_chart(build_cost_by_width(list(C.MONO_WIDTHS), costs, a.ucs.cost), width="stretch", key="mono_curve")
if network == "grid":
    if st.button("Nicht-Monotonie über 50 feste Instanzen messen", key="mono_start"):
        st.session_state["mono_done"] = st.session_state.get("mono_done", set()) | {replace(settings, seed=0, width=C.DEFAULT_WIDTH)}
    key_mono = replace(settings, seed=0, width=C.DEFAULT_WIDTH)
    if key_mono in st.session_state.get("mono_done", set()):
        with st.spinner("Rechne 50 Instanzen x 24 Breiten..."):
            mono = _monotonicity(key_mono)
        c1, c2, c3 = st.columns(3)
        c1.metric("Instanzen mit schlechterem breiteren Strahl", f"{mono['non_monotone_share']:.0f} %", delta=f"von {mono['n']}", delta_color="off")
        c2.metric("... davon: breiter scheitert", f"{mono['wider_fails_share']:.0f} %", delta="schmalerer fand das Ziel", delta_color="off")
        c3.metric("... davon: breiter kostet mehr", f"{mono['wider_costlier_share']:.0f} %", delta="höhere endliche Kosten", delta_color="off")
        if mono["example_seeds"]:
            st.caption("Beispiel-Seeds mit Nicht-Monotonie (in die Seitenleiste eintragen): " + ", ".join(str(s) for s in mono["example_seeds"][:8]))
        st.caption(f"Instanzen: Seeds 200000–200049, Rastergröße {settings.side}, Hindernisdichte {settings.obstacle_pct} %, Rangfolge {C.RANK_LABELS[settings.rank]}.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Mehr Breite ist besser** | Im Median ja: die Lücke sinkt von 16 % (Breite 1) auf 0 % (ab Breite 5), ab Breite 8 waren alle 5 Instanzen optimal. Je Instanz nicht garantiert: bei 6 % der Instanzen (Größe 12, 15 % Hindernisse) ist irgendein breiterer Strahl schlechter, bei Größe 20 bei 14 %. | **Monobeam** (Lemons et al. 2022) setzt genau an dieser Nicht-Monotonie an |
| **Die Suche findet immer eine Lösung** | Kein Zurück: ein schmaler Strahl kann das Ziel verlieren. Bei 15 % Hindernissen scheitern 20 % der Läufe (Breite 1-3), bei 40 % Hindernissen 60 % (Breite 1) und 40 % (Breite 2). Bei 40 % Hindernissen scheitert teils sogar ein breiterer Strahl, obwohl ein schmalerer das Ziel fand (10 % der Instanzen). Greedy Best-First mit Backtracking scheitert in diesen Läufen nie. | **Beam Stack Search** (Beam Search + Backtracking, Zhou & Hansen 2005) |
| **Der Strahl ist billiger als A\\*** | Nur wenn man Lücke oder Scheitern in Kauf nimmt: bis Breite 4 unter A\\* (0.27x bis 0.78x der Expansionen), ab Breite 6 darüber (1.08x), bei Breite 8 1.24x. Zuverlässig optimal war er erst bei Breite 8 - dort teurer als A\\*. | A\\* / IDA\\* (optimal, ohne Breitenwahl) |
| **Der Strahl spart Speicher** | Die hier gezählten Knoten sind alle entdeckten (wie bei A\\*, mit Closed-Set zur Duplikaterkennung). Ein echter Speicher-Strahl ohne Closed-Set bräuchte nur die Spitzenbreite (Median 6 bei Breite 3) - diese Variante ist hier nicht gebaut. | (nicht gebaut) |
| **Unbegrenzte Breite = optimal** | Die Schicht-Suche findet den Pfad mit den wenigsten KANTEN. Auf diesen Rastern war er in allen getesteten Instanzen trotzdem optimal; ein konstruiertes Gegenbeispiel (Direktkante 10 gegen Umweg 3) zeigt, dass es allgemein nicht gilt. | Uniform-Cost / A\\* |
| **Synthetische Instanzen** | Ein Raster mit Jitter, Vierer-Nachbarschaft, keine Zeitfenster, keine gerichteten Kanten. Andere Graphstrukturen wurden nicht gemessen. | Echte Straßennetze (hier nicht gebaut) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Strahl.** $B_0 = \{s\}$. Kandidaten $C_{t+1} = \{ v \notin \text{Expandiert} : (u,v) \in E,\ u \in B_t \}$ mit
$g(v) = \min_{u \in B_t} g(u) + w(u,v)$. Rangfolge $r(v) = h(v)$ oder $r(v) = g(v) + h(v)$;
$B_{t+1}$ = die $k$ Kandidaten mit kleinstem $r$ (Tie-Break Knotenindex). Ende, sobald das Ziel in $C_{t+1}$ liegt
(kleinstes $g$); scheitert, wenn $C_{t+1} = \emptyset$.

**Sonderfälle, im Test nachgewiesen.** $k = 1$ mit $r = h$ ist ein gieriger Abstieg (gegen eine unabhängig
geschriebene Referenz geprüft); bei Einheitsgewichten, $h \equiv 0$ und unbegrenzter Breite ist es Breitensuche.

**Lücke.** $100 \cdot (c_\text{Beam} - c^*) / c^*$ mit $c^*$ aus Uniform-Cost, nur über gelöste Läufe; daneben stets die
Scheiter-Quote.

**Nicht-Monotonie.** Eine Instanz heißt nicht-monoton, wenn es Breiten $k < k'$ mit $c(k') > c(k)$ gibt
($c = \infty$ bei Scheitern).

**Literatur.** Beam Search geht auf B. Lowerre (1976), *The HARPY Speech Recognition System* (Dissertation,
Carnegie Mellon University) zurück.

Implementiert in `beam_algorithm.py` (Suchkerne aus der A*-Demo, `beam_search` neu), `beam_graph.py`/`beam_scenario.py`
(Graph, Raster- und Fallen-Instanzen, Kopie), `beam_evaluation.py` (Kennzahlen, Sweeps, Nicht-Monotonie).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
