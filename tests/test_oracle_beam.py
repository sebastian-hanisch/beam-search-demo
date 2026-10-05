"""Unabhängiges Orakel für Beam Search: (1) eine eigene Neuimplementierung aus der Spezifikation (Schichten, geschlossene
Menge, Beschneiden auf die besten k) muss für jede Breite und beide Rangfolgen Pfad, Scheitern, Expansionen, Speicher,
Schichtzahl und Spitzenbreite identisch liefern; (2) bei unbegrenzter Breite muss das Ergebnis der billigste Pfad unter
allen Pfaden mit minimaler Kantenzahl sein (Orakel: networkx-BFS-Schichten + Programmierung über die Schichten);
(3) Kosten nie unter dem Brute-Force-Optimum, auch bei Gewichten, die die Heuristik unzulässig machen."""

import math
import random

import numpy as np
import pytest

import beam_algorithm as A
import beam_graph as G
import beam_scenario as S

nx = pytest.importorskip("networkx")


def _to_nx(g):
    H = nx.Graph()
    H.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.neighbors[u], g.weights[u]):
            H.add_edge(u, v, weight=w)
    return H


def _ref_beam(H, xy, s, t, width, rank):
    hh = {v: math.hypot(*(xy[v] - xy[t])) for v in H.nodes}
    gv, par, done, doneset, seen, layer, layers, peak = {s: 0.0}, {s: None}, [], set(), {s}, [s], [], 1
    while True:
        for u in layer:
            done.append(u)
            doneset.add(u)
        cand = {}
        for u in layer:
            for v in H.neighbors(u):
                if v in doneset:
                    continue
                val = gv[u] + H[u][v]["weight"]
                if v not in cand or val < cand[v][0]:
                    cand[v] = (val, u)
        seen |= set(cand)
        peak = max(peak, len(cand))
        layers.append(list(layer))
        if t in cand:
            path, c = [t], cand[t][1]
            while c is not None:
                path.append(c)
                c = par[c]
            return dict(path=path[::-1], cost=cand[t][0], failed=False, done=done, stored=len(seen), layers=layers, peak=peak)
        if not cand:
            return dict(path=[], cost=math.inf, failed=True, done=done, stored=len(seen), layers=layers, peak=peak)
        key = (lambda v: (hh[v], v)) if rank == "h" else (lambda v: (cand[v][0] + hh[v], v))
        layer = sorted(cand, key=key)[:width]
        for v in layer:
            gv[v], par[v] = cand[v]


def _cheapest_among_fewest_edges(H, s, t):
    dist = nx.single_source_shortest_path_length(H, s)
    best = {s: 0.0}
    for level in range(1, dist[t] + 1):
        for v in (x for x in H.nodes if dist.get(x) == level):
            options = [best[u] + H[u][v]["weight"] for u in H.neighbors(v) if dist.get(u) == level - 1 and u in best]
            if options:
                best[v] = min(options)
    return dist[t], best[t]


def _brute_force_min_cost(H, s, t):
    best, stack = math.inf, [(s, {s}, 0.0)]
    while stack:
        u, vis, c = stack.pop()
        if u == t:
            best = min(best, c)
            continue
        stack.extend((v, vis | {v}, c + H[u][v]["weight"]) for v in H.neighbors(u) if v not in vis)
    return best


@pytest.mark.parametrize("seed", range(25))
def test_beam_search_matches_reference_implementation_on_grids(seed):
    rnd = random.Random(seed)
    side, pct = rnd.randint(4, 10), rnd.choice([0, 15, 30, 40])
    inst = S.grid_instance(side, pct, rnd.randint(0, 999999))
    g, s, t = inst.graph, inst.start, inst.goal
    H = _to_nx(g)
    for width in (1, 2, 3, 6, 10**6):
        for rank in ("h", "f"):
            r, o = A.beam_search(g, s, t, width, rank), _ref_beam(H, g.xy, s, t, width, rank)
            assert (r.failed, r.path, r.order, r.stored, r.peak_width) == (o["failed"], o["path"], o["done"], o["stored"], o["peak"])
            assert [list(b) for b, _ in r.per_layer] == o["layers"]
            if not r.failed:
                assert r.cost == pytest.approx(o["cost"], abs=1e-9)
        hops, best = _cheapest_among_fewest_edges(H, s, t)
        r = A.beam_search(g, s, t, 10**6, "h")
        assert len(r.path) - 1 == hops and r.cost == pytest.approx(best, abs=1e-9)


@pytest.mark.parametrize("seed", range(30))
def test_beam_search_on_small_random_graphs_never_beats_brute_force(seed):
    rnd = random.Random(1000 + seed)
    n = rnd.randint(3, 9)
    xy = np.array([[rnd.uniform(0, 10), rnd.uniform(0, 10)] for _ in range(n)])
    edges = [(i, j, rnd.uniform(0.5, 20)) for i in range(n) for j in range(i + 1, n) if rnd.random() < 0.45]
    g = G.from_edges(n, xy, edges)
    s, t = rnd.sample(range(n), 2)
    H = _to_nx(g)
    reachable = nx.has_path(H, s, t)
    best = _brute_force_min_cost(H, s, t)
    for width in (1, 2, 4, 100):
        for rank in ("h", "f"):
            r, o = A.beam_search(g, s, t, width, rank), _ref_beam(H, g.xy, s, t, width, rank)
            assert (r.failed, r.path) == (o["failed"], o["path"])
            if not reachable:
                assert r.failed and r.path == [] and r.cost == math.inf
            elif not r.failed:
                assert r.cost >= best - 1e-9
    if reachable:
        r = A.beam_search(g, s, t, 100, "h")
        hops, cheapest = _cheapest_among_fewest_edges(H, s, t)
        assert len(r.path) - 1 == hops and r.cost == pytest.approx(cheapest, abs=1e-9)
