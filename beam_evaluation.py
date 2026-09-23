"""Auswertung: was bringt ein breiterer Strahl, und was kostet der fehlende Rückweg? Vergleich von Beam Search mit
Greedy Best-First (Wurzel: ein Kandidat, mit Backtracking), A* und Uniform-Cost-Search (Referenz) auf demselben Graphen.

Kennzahlen (alle über die 5 festen Sweep-Instanzen, Seeds 100000-100004, deterministisch):

- **Optimalitätslücke** = 100 * (Kosten Beam - Kosten UCS) / Kosten UCS - nur über GELÖSTE Läufe (ein gescheiterter Lauf
  hat keine Kosten). Deshalb steht daneben immer die **Scheiter-Quote** und der **Optimal-Anteil** (Anteil ALLER Läufe mit
  Lücke 0; gescheiterte zählen als nicht optimal) - ein Median nur über die Überlebenden täuscht sonst bei schmalem Strahl.
- **Expansions-Verhältnis** = Expansionen Beam / Expansionen A*, **Speicher-Verhältnis** = gespeicherte Knoten Beam /
  gespeicherte Knoten A* (beide nur über gelöste Läufe; < 1 heißt: der Strahl schaut sich weniger an als A*).
- **Nicht-Monotonie** (`monotonicity`): Anteil Instanzen, bei denen irgendein BREITERER Strahl schlechter ist als ein
  schmalerer (höhere Kosten, oder Scheitern trotz Erfolg des schmaleren) - über die Breiten 1..24.

Median mit Spannweite (Minimum bis Maximum), weil die Lücken schief verteilt sind."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import beam_algorithm as A
import beam_constants as C
import beam_scenario as S


@dataclass(frozen=True)
class Settings:
    network: str = "grid"           # "grid" oder "trap"
    side: int = C.DEFAULT_SIDE
    obstacle_pct: int = C.DEFAULT_OBSTACLE
    seed: int = C.DEFAULT_SEED
    width: int = C.DEFAULT_WIDTH
    rank: str = C.DEFAULT_RANK


@lru_cache(maxsize=512)
def instance(side, obstacle_pct, seed):
    return S.grid_instance(side, obstacle_pct, seed)


def _instance(settings):
    return S.trap_instance() if settings.network == "trap" else instance(settings.side, settings.obstacle_pct, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    beam: A.BeamResult
    gbfs: A.SearchResult
    astar: A.SearchResult
    ucs: A.SearchResult

    @property
    def solved(self):
        return not self.beam.failed

    @property
    def gap(self):
        """Optimalitätslücke des Strahls in %; NaN bei gescheitertem Lauf."""
        return 100.0 * (self.beam.cost - self.ucs.cost) / self.ucs.cost if self.solved else float("nan")

    @property
    def gbfs_gap(self):
        return 100.0 * (self.gbfs.cost - self.ucs.cost) / self.ucs.cost

    @property
    def optimal(self):
        return self.solved and self.beam.cost <= self.ucs.cost + 1e-9

    @property
    def expansion_ratio(self):
        """Expansionen Beam / Expansionen A*; NaN bei gescheitertem Lauf."""
        return self.beam.expansions / self.astar.expansions if self.solved else float("nan")

    @property
    def stored_ratio(self):
        return self.beam.stored / self.astar.stored if self.solved else float("nan")


def analyse(settings):
    inst = _instance(settings)
    beam = A.beam_search(inst.graph, inst.start, inst.goal, settings.width, settings.rank)
    gbfs = A.greedy_best_first(inst.graph, inst.start, inst.goal)
    astar = A.a_star(inst.graph, inst.start, inst.goal)
    ucs = A.uniform_cost_search(inst.graph, inst.start, inst.goal)
    return Analysis(settings, inst, beam, gbfs, astar, ucs)


# --- Sweeps --------------------------------------------------------------------------------------------------------------------------------------


def _median_range(values):
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.min(values)), float(np.max(values))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=seed)) for seed in seeds]
    n = len(rows)
    out = {
        "n_runs": n,
        "n_solved": sum(r.solved for r in rows),
        "failed_share": 100.0 * sum(not r.solved for r in rows) / n,
        "optimal_share": 100.0 * sum(r.optimal for r in rows) / n,
    }
    for key, values in (
        ("gap", [r.gap for r in rows]),                          # nur gelöste Läufe
        ("gbfs_gap", [r.gbfs_gap for r in rows]),
        ("expansion_ratio", [r.expansion_ratio for r in rows]),
        ("stored_ratio", [r.stored_ratio for r in rows]),
        ("peak_width", [float(r.beam.peak_width) for r in rows]),
    ):
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _median_range(values)
    out["beam_expansions"] = float(np.median([r.beam.expansions for r in rows]))
    out["astar_expansions"] = float(np.median([r.astar.expansions for r in rows]))
    out["ucs_expansions"] = float(np.median([r.ucs.expansions for r in rows]))
    out["gbfs_expansions"] = float(np.median([r.gbfs.expansions for r in rows]))
    return out


SWEEP_VALUES = {"width": C.WIDTHS, "obstacle_pct": C.OBSTACLE_SWEEP, "side": C.SCALING_SIDES}
SWEEP_LABELS = {"width": "Strahlbreite k", "obstacle_pct": "Hindernisdichte (%)", "side": "Rastergröße (Seitenlänge)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


# --- Nicht-Monotonie -----------------------------------------------------------------------------------------------------------------------------


def cost_by_width(settings, widths=C.MONO_WIDTHS):
    """Kosten (inf = gescheitert) für jede Breite auf der Instanz von `settings` (die Breite in `settings` wird ignoriert)."""
    inst = _instance(settings)
    return [A.beam_search(inst.graph, inst.start, inst.goal, k, settings.rank).cost for k in widths]


def is_non_monotone(costs):
    """(irgendwo schlechter, davon: ein breiterer Strahl SCHEITERT trotz Erfolg eines schmaleren, davon: ein breiterer
    Strahl hat höhere endliche Kosten). Schlechter = ein breiterer Strahl hat höhere Kosten (inf zählt) als ein schmalerer."""
    worse = fail = cost = False
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            if costs[j] > costs[i] + 1e-9:
                worse = True
                if costs[j] == float("inf"):
                    fail = True
                else:
                    cost = True
    return worse, fail, cost


def monotonicity(base, seeds=C.MONO_SEEDS, widths=C.MONO_WIDTHS, **changes):
    """Anteil (in %) der Instanzen, bei denen ein breiterer Strahl schlechter ist - über `seeds` und `widths`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        costs = cost_by_width(replace(s0, seed=seed), widths)
        rows.append((seed, costs, *is_non_monotone(costs)))
    n = len(rows)
    return {
        "n": n,
        "non_monotone_share": 100.0 * sum(r[2] for r in rows) / n,
        "wider_fails_share": 100.0 * sum(r[3] for r in rows) / n,
        "wider_costlier_share": 100.0 * sum(r[4] for r in rows) / n,
        "example_seeds": [r[0] for r in rows if r[2]],
        "rows": rows,
    }
