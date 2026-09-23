"""Die zentrale Korrektheits-Kette für Beam Search: gültiger Pfad, Kosten nie unter dem Optimum (gegen Brute-Force),
drei Sonderfälle (Kettengraph, Einheitsgewichte + h = 0 + unbegrenzte Breite = Breitensuche, Breite 1 = gieriger
Abstieg gegen eine unabhängige Referenz), Schicht-/Duplikat-Buchführung, Scheitern, und die Grenze der Schicht-Suche
(unbegrenzte Breite findet die wenigsten KANTEN, nicht die kleinsten Kosten)."""

import numpy as np
import pytest

import beam_algorithm as A
import beam_graph as G
import beam_scenario as S

EPS = 1e-9
HUGE = 10 ** 6
RANKS = ("h", "f")


def _brute_force_shortest_cost(graph, start, goal):
    best = None
    stack = [(start, [start], 0.0)]
    while stack:
        node, path, cost = stack.pop()
        if node == goal:
            if best is None or cost < best:
                best = cost
            continue
        for v, w in zip(graph.neighbors[node], graph.weights[node]):
            if v not in path:
                stack.append((v, path + [v], cost + w))
    return best


def _greedy_descent_reference(graph, start, goal):
    """Unabhängig geschriebener gieriger Abstieg mit Erledigt-Menge: vom aktuellen Knoten zum noch nicht besuchten
    Nachbarn mit kleinstem h (Tie-Break Knotenindex); Ziel unter den Nachbarn -> fertig; keine Nachbarn -> gescheitert."""
    h = A.heuristic(graph.xy, goal)
    path, done, cur = [start], {start}, start
    while True:
        nbrs = [v for v in graph.neighbors[cur] if v not in done]
        if goal in nbrs:
            return path + [goal]
        if not nbrs:
            return None
        cur = min(nbrs, key=lambda v: (h[v], v))
        done.add(cur)
        path.append(cur)


# --- Gültigkeit und Untergrenze ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("seed", range(8))
def test_beam_returns_a_valid_simple_path_with_recomputed_cost(seed, width, rank):
    inst = S.grid_instance(side=6, obstacle_pct=20, seed=seed)
    result = A.beam_search(inst.graph, inst.start, inst.goal, width, rank)
    if result.failed:
        assert result.path == [] and result.cost == float("inf")
        return
    assert result.path[0] == inst.start and result.path[-1] == inst.goal
    assert len(set(result.path)) == len(result.path)
    for u, v in zip(result.path[:-1], result.path[1:]):
        assert v in inst.graph.neighbors[u]
    assert G.path_cost(inst.graph, result.path) == pytest.approx(result.cost, abs=1e-6)


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("width", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", range(12))
def test_beam_never_beats_the_brute_force_optimum(seed, width, rank):
    inst = S.grid_instance(side=5, obstacle_pct=15, seed=seed)
    result = A.beam_search(inst.graph, inst.start, inst.goal, width, rank)
    if not result.failed:
        assert result.cost >= _brute_force_shortest_cost(inst.graph, inst.start, inst.goal) - 1e-6


@pytest.mark.parametrize("rank", RANKS)
def test_beam_on_the_trap_is_valid_and_never_below_the_optimum(rank):
    inst = S.trap_instance()
    optimum = _brute_force_shortest_cost(inst.graph, inst.start, inst.goal)
    for width in (1, 2, 3, HUGE):
        result = A.beam_search(inst.graph, inst.start, inst.goal, width, rank)
        assert not result.failed and result.cost >= optimum - 1e-6


def test_beam_is_deterministic():
    inst = S.grid_instance(side=8, obstacle_pct=20, seed=3)
    r1 = A.beam_search(inst.graph, inst.start, inst.goal, 3, "h")
    r2 = A.beam_search(inst.graph, inst.start, inst.goal, 3, "h")
    assert r1.path == r2.path and r1.order == r2.order and r1.per_layer == r2.per_layer


def test_invalid_arguments_are_rejected():
    inst = S.trap_instance()
    with pytest.raises(ValueError):
        A.beam_search(inst.graph, inst.start, inst.goal, 0)
    with pytest.raises(ValueError):
        A.beam_search(inst.graph, inst.start, inst.goal, 2, "x")


# --- Sonderfälle -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 3, HUGE])
@pytest.mark.parametrize("rank", RANKS)
def test_chain_graph_is_solved_by_every_width_with_one_expansion_per_node(width, rank):
    n = 6
    graph = G.from_edges(n, [(i, 0) for i in range(n)], [(i, i + 1, 1.0) for i in range(n - 1)])
    result = A.beam_search(graph, 0, n - 1, width, rank)
    assert result.path == list(range(n)) and result.cost == n - 1
    assert result.expansions == n - 1 and result.stored == n and result.layers == n - 1


@pytest.mark.parametrize("seed", range(20))
def test_unit_weights_zero_heuristic_and_unlimited_width_is_breadth_first_search(seed):
    """Alle Koordinaten gleich -> h == 0, Einheitsgewichte: jede Schicht bleibt vollständig erhalten = Breitensuche,
    Kosten == kürzeste Kantenzahl == Uniform-Cost-Optimum."""
    rng = np.random.default_rng(seed)
    n = 12
    edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    edges += [(int(a), int(b), 1.0) for a, b in rng.integers(0, n, size=(10, 2)) if a != b]
    graph = G.from_edges(n, [(0, 0)] * n, edges)
    result = A.beam_search(graph, 0, n - 1, HUGE, "h")
    assert not result.failed
    assert result.cost == pytest.approx(A.uniform_cost_search(graph, 0, n - 1).cost, abs=EPS)
    assert result.cost == pytest.approx(_brute_force_shortest_cost(graph, 0, n - 1), abs=EPS)


@pytest.mark.parametrize("seed", range(40))
def test_width_one_is_the_independent_greedy_descent(seed):
    inst = S.grid_instance(side=7, obstacle_pct=(seed % 5) * 10, seed=seed)
    reference = _greedy_descent_reference(inst.graph, inst.start, inst.goal)
    result = A.beam_search(inst.graph, inst.start, inst.goal, 1, "h")
    if reference is None:
        assert result.failed and result.path == []
    else:
        assert not result.failed and result.path == reference


def test_start_equals_goal_is_solved_without_expansion():
    inst = S.trap_instance()
    result = A.beam_search(inst.graph, inst.start, inst.start, 2)
    assert result.path == [inst.start] and result.cost == 0.0 and result.expansions == 0 and not result.failed


# --- Schicht- und Duplikat-Buchführung -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("width", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", range(10))
def test_layer_bookkeeping_is_consistent(seed, width, rank):
    inst = S.grid_instance(side=8, obstacle_pct=15, seed=seed)
    result = A.beam_search(inst.graph, inst.start, inst.goal, width, rank)
    beams = [set(b) for b, _dropped in result.per_layer]
    assert all(len(b) <= width for b in beams)
    assert sum(len(b) for b in beams) == result.expansions == len(result.order)
    assert len(set(result.order)) == len(result.order)                 # kein Knoten zweimal expandiert
    assert result.layers == len(result.per_layer)
    assert result.expansions <= result.stored <= inst.graph.n
    assert result.peak_width >= max(len(b) for b in beams)
    if not result.failed:
        assert len(result.path) - 1 == result.layers                   # Ziel wird in Schicht L erzeugt -> Pfad hat L Kanten
    for (beam, dropped) in result.per_layer:
        assert not set(beam) & set(dropped)


# --- Scheitern -------------------------------------------------------------------------------------------------------------------------------


def test_unreachable_goal_always_fails_and_claims_no_path():
    graph = G.from_edges(4, [(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 1, 1.0), (2, 3, 1.0)])
    for width in (1, 3, HUGE):
        result = A.beam_search(graph, 0, 3, width)
        assert result.failed and result.path == [] and result.cost == float("inf")


def test_a_narrow_beam_can_lose_the_goal_that_a_wider_beam_finds():
    """Scheitern nach Erfolg mit breiterem Strahl ist die harte Form der Unvollständigkeit: gesucht wird eine
    tatsächliche Instanz aus den festen Testseeds (kein Vorab-Versprechen)."""
    found = False
    for seed in range(200000, 200040):
        inst = S.grid_instance(side=12, obstacle_pct=40, seed=seed)
        narrow = A.beam_search(inst.graph, inst.start, inst.goal, 1, "h")
        wide = A.beam_search(inst.graph, inst.start, inst.goal, 8, "h")
        if narrow.failed and not wide.failed:
            found = True
            assert narrow.path == [] and wide.path[-1] == inst.goal
            break
    assert found


# --- Grenze der Schicht-Suche -----------------------------------------------------------------------------------------------------------------


def test_unlimited_width_finds_the_fewest_edges_not_the_cheapest_path():
    """Direktkante 0-3 kostet 10, der Umweg 0-1-2-3 nur 3: die Schicht-Suche erzeugt das Ziel schon in Schicht 1 und
    liefert 10 - trotz unbegrenzter Breite. Optimalität gilt nur im Einheitsgewicht-Sonderfall (Test oben)."""
    xy = [(0, 0), (1, 0), (2, 0), (3, 0)]
    graph = G.from_edges(4, xy, [(0, 3, 10.0), (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)])
    result = A.beam_search(graph, 0, 3, HUGE, "f")
    assert result.cost == 10.0 and result.layers == 1
    assert A.uniform_cost_search(graph, 0, 3).cost == 3.0


# --- Geerbte Eigenschaften der kopierten Suchkerne -------------------------------------------------------------------------------------------


def test_inherited_heuristic_is_admissible_and_the_copied_searches_are_ordered():
    for seed in range(10):
        inst = S.grid_instance(side=7, obstacle_pct=20, seed=seed)
        h = A.heuristic(inst.graph.xy, inst.goal)
        ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
        for node in range(inst.graph.n):
            assert h[node] <= A.uniform_cost_search(inst.graph, node, inst.goal).cost + EPS
        assert A.a_star(inst.graph, inst.start, inst.goal).cost == pytest.approx(ucs.cost, abs=EPS)
        assert A.greedy_best_first(inst.graph, inst.start, inst.goal).cost >= ucs.cost - EPS
