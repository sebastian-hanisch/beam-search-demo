"""Suchkerne - `_search`, `greedy_best_first`, `uniform_cost_search`, `a_star` wortgleich aus `astar-demo`/`ida-star-demo`
(dort korrektheitsgeprüft; Vergleichsgrößen dieses Stücks) und NEU `beam_search`.

Beam Search: schichtweise Suche (Schicht = Kantenzahl vom Start). Je Schicht werden ALLE Strahl-Knoten expandiert, die
Nachfolger gesammelt (je Knoten der kleinste g-Wert = Duplikaterkennung innerhalb der Schicht; schon expandierte
Knoten früherer Schichten kommen nicht wieder auf, sonst Zyklen) und nach der Rangfolge (`h` oder `g + h`, Tie-Break
Knotenindex) auf die besten `width` beschnitten. KEIN Zurück: was aus dem Strahl fällt, ist verloren. Wird das Ziel
erzeugt, endet die Suche mit dem kleinsten g unter den Zielkandidaten dieser Schicht; eine leere Kandidatenmenge heißt
`failed` (kein Pfad behauptet). Bei unbegrenzter Breite findet diese Schicht-Suche den Pfad mit den WENIGSTEN KANTEN,
nicht zwingend den billigsten. Deterministisch: feste Nachbarreihenfolge, Tie-Break nach Knotenindex.
"""

import heapq
from dataclasses import dataclass, field

import numpy as np


def heuristic(xy, goal):
    """Euklidischer Abstand jedes Knotens zum Ziel - vektorisiert. Bei echten Kantengewichten (siehe
    `beam_scenario.py`) automatisch zulässig (Dreiecksungleichung)."""
    return np.hypot(*(xy - xy[goal]).T)


@dataclass
class SearchResult:
    path: list                  # Knotenfolge Start..Ziel, oder [] falls kein Pfad existiert
    cost: float                 # Summe der Kantengewichte entlang des Pfades
    expansions: int             # Zahl der expandierten Knoten (Effizienz-Kennzahl dieses Stücks)
    order: list = field(default_factory=list)     # Reihenfolge der expandierten Knoten (für die Schritt-Visualisierung)
    stored: int = 0             # Speicher-Kennzahl: gespeicherte Knoten am Ende (A*/UCS/GBFS: entdeckte Knoten; IDA*: max. Pfadtiefe)


def _search(graph, start, goal, priority_fn, relax=True):
    """`priority_fn(node, g_cost) -> float` bestimmt die Warteschlangen-Priorität. `g_cost` ist der bislang
    aufgelaufene Pfadwert zu `node` (für Uniform-Cost-Search gebraucht, von Greedy Best-First ignoriert).

    `relax`: ob ein noch nicht expandierter, aber schon entdeckter Knoten einen GÜNSTIGEREN Elternknoten
    bekommt, sobald ein billigerer Weg zu ihm gefunden wird (klassische Dijkstra-Relaxation - für
    Uniform-Cost-Search nötig, damit es tatsächlich optimal bleibt). Bei `relax=False` behält ein Knoten für
    immer den ERSTEN gefundenen Elternknoten (echtes "kein Backtracking" - der Kern der GBFS-Schwäche: eine
    Relaxation hier würde den gemessenen Qualitätsverlust künstlich kleinrechnen, da GBFS dann doch beiläufig
    von g(n) profitieren würde, obwohl es g(n) laut Definition komplett ignoriert)."""
    counter = 0
    frontier = [(priority_fn(start, 0.0), counter, start, 0.0)]
    came_from = {start: None}
    g_cost = {start: 0.0}
    visited = set()
    order = []

    while frontier:
        _priority, _c, node, g = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return SearchResult(path, g, len(order), order, len(g_cost))
        for v, w in zip(graph.neighbors[node], graph.weights[node]):
            if v in visited:
                continue
            g_v = g + w
            is_new = v not in g_cost
            if is_new or (relax and g_v < g_cost[v]):
                g_cost[v] = g_v
                came_from[v] = node
                counter += 1
                heapq.heappush(frontier, (priority_fn(v, g_v), counter, v, g_v))

    return SearchResult([], float("inf"), len(order), order, len(g_cost))


def greedy_best_first(graph, start, goal):
    h = heuristic(graph.xy, goal)
    return _search(graph, start, goal, lambda node, g: h[node], relax=False)


def uniform_cost_search(graph, start, goal):
    return _search(graph, start, goal, lambda node, g: g, relax=True)


def a_star(graph, start, goal):
    h = heuristic(graph.xy, goal)
    return _search(graph, start, goal, lambda node, g: g + h[node], relax=True)


@dataclass
class BeamResult(SearchResult):
    failed: bool = False
    layers: int = 0
    per_layer: list = field(default_factory=list)    # [(Strahl, verworfene Kandidaten), ...] je Schicht
    peak_width: int = 0                              # größte Kandidatenzahl einer Schicht vor dem Beschneiden


def beam_search(graph, start, goal, width, rank="h"):
    if width < 1:
        raise ValueError("width muss mindestens 1 sein")
    if rank not in ("h", "f"):
        raise ValueError("rank muss 'h' oder 'f' sein")
    h = heuristic(graph.xy, goal)
    neighbors, weights = graph.neighbors, graph.weights
    beam = [start]
    g = {start: 0.0}
    parent = {start: None}
    expanded = set()
    discovered = {start}
    order = []
    per_layer = []
    peak = 1

    def build_path(node):
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    if start == goal:
        return BeamResult([start], 0.0, 0, [], 1, False, 0, [], 1)

    while True:
        cand_g, cand_parent = {}, {}
        current = list(beam)
        for node in beam:
            expanded.add(node)
            order.append(node)
        for node in beam:
            for v, w in zip(neighbors[node], weights[node]):
                if v in expanded:
                    continue
                g_v = g[node] + w
                if v not in cand_g or g_v < cand_g[v]:
                    cand_g[v] = g_v
                    cand_parent[v] = node
        discovered.update(cand_g)
        peak = max(peak, len(cand_g))
        if goal in cand_g:
            parent[goal] = cand_parent[goal]
            per_layer.append((current, []))
            return BeamResult(build_path(goal), cand_g[goal], len(order), order, len(discovered), False, len(per_layer), per_layer, peak)
        if not cand_g:
            per_layer.append((current, []))
            return BeamResult([], float("inf"), len(order), order, len(discovered), True, len(per_layer), per_layer, peak)
        if rank == "h":
            ranked = sorted(cand_g, key=lambda v: (h[v], v))
        else:
            ranked = sorted(cand_g, key=lambda v: (cand_g[v] + h[v], v))
        beam, dropped = ranked[:width], ranked[width:]
        per_layer.append((current, dropped))
        for v in beam:
            g[v] = cand_g[v]
            parent[v] = cand_parent[v]
