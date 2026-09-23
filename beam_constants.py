"""Konstanten der Beam-Search-Demo: Raster-Geometrie (wortgleich zur A*-Demo), Regler, gemessene Werte, Presets."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
JITTER = 0.35                    # Lageabweichung je Zelle, Anteil des Zellenabstands

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE, SIDE_STEP = 5, 25, 12, 1     # Rastergröße (Zellen je Kante)
OBSTACLE_MIN, OBSTACLE_MAX, DEFAULT_OBSTACLE, OBSTACLE_STEP = 0, 40, 15, 5   # Prozent gesperrte Zellen
SEED_MAX = 999999
DEFAULT_SEED = 35

WIDTHS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)     # Strahlbreiten des Reglers und des Breiten-Sweeps
DEFAULT_WIDTH = 3                              # MUSS Mitglied von WIDTHS sein (st.select_slider snappt sonst still)
RANKS = ("h", "f")                             # Rangfolge: nur Heuristik (Kind der Wurzel) oder g + h
DEFAULT_RANK = "h"
RANK_LABELS = {"h": "h (nur Heuristik)", "f": "f = g + h"}

SWEEP_SEEDS = tuple(range(100000, 100005))
SCALING_SIDES = (6, 10, 14, 18, 22)
OBSTACLE_SWEEP = (0, 10, 20, 30, 40)
MONO_SEEDS = tuple(range(200000, 200050))      # 50 feste Instanzen für die Nicht-Monotonie-Rate
MONO_WIDTHS = tuple(range(1, 25))

# --- Gemessene Werte (MEDIAN über 5 feste Sweep-Instanzen, Seeds 100000-100004; Rastergröße 12, Hindernisdichte 15 %,
# --- Rangfolge h, sofern nicht anders angegeben; 2026-09-23, alle Werte über ev.run_config/ev.sweep/ev.monotonicity
# --- nachgerechnet, s. tests/test_claims.py). Lücke und Verhältnisse nur über GELÖSTE Läufe - daneben stets
# --- Scheiter-Quote und Optimal-Anteil (Anteil ALLER Läufe mit Lücke 0), sonst täuscht der Median der Überlebenden. ---
# ZENTRALE FRAGE - macht mehr Breite das Ergebnis besser? Im Mittel JA: Lücke 16.2/7.3/2.8/0.35/0.0/0.0/0.0 % bei
#   Breite 1/2/3/4/5/6/8; Optimal-Anteil 0/0/20/40/60/60/100 %; Scheitern 20 % bei Breite 1-3, danach 0. Ab Breite 8
#   waren alle 5 Instanzen optimal. GBFS (Wurzel, mit Backtracking) hat 14.7 % Lücke und scheitert nie.
# ABER: nicht je Instanz monoton. Über 50 feste Instanzen (Breiten 1-24): bei Größe 12 / 15 % Hindernissen ist bei 6 %
#   der Instanzen irgendein breiterer Strahl schlechter (höhere Kosten), bei 0 % Hindernissen 2 %, bei Größe 20 14 %,
#   und bei 40 % Hindernissen 10 % - dort in Form von SCHEITERN: ein breiterer Strahl verliert das Ziel, obwohl ein
#   schmalerer es fand (z. B. Seed 200007: Breite 1 findet es, Breite 2 scheitert, ab Breite 3 klappt es wieder).
# EFFIZIENZ GEGEN A* (Expansions-Verhältnis Beam / A*): Breite 1-4: 0.27/0.47/0.64/0.78 (weniger als A*), Breite 5: 0.97,
#   ab Breite 6 MEHR als A* (1.08), Breite 8: 1.24, bei unbegrenzter Breite gesättigt bei ~1.25 (Uniform-Cost expandiert
#   im Median 121, A* 93, der breite Strahl 120). Speicher-Verhältnis (gespeicherte Knoten): 0.45/0.60/0.71/0.85/0.94/1.00/
#   1.10. Der Strahl ist NUR dann billiger als A*, wenn man Lücke oder Scheitern in Kauf nimmt; zuverlässig optimal
#   (5 von 5) ist er erst bei Breite 8 - dort teurer als A*.
# RANGFOLGE f = g + h statt h: bei Breite 2-4 kleinere Lücke (4.5/0.35/0.0 gegen 7.3/2.8/0.35 %) und bei Breite 2 keine
#   Ausfälle (0 gegen 20 %) - aber bei Breite 1 MEHR Ausfälle (40 gegen 20 %, nur 5 Instanzen). Nicht-Monotonie bleibt.
# HINDERNISDICHTE (Breite 3, 0/10/20/30/40 %): Lücke 2.7/0.0/1.3/0.0/0.0 %, Scheiter-Quote 0/0/20/0/0 %, Expansions-
#   Verhältnis 0.61/0.72/0.82/0.98/1.00 - der Vorteil gegen A* schmilzt mit den Hindernissen (weniger Spielraum, der
#   Strahl muss fast alles ansehen). Bei 40 % Hindernissen scheitert der schmale Strahl oft: Scheiter-Quote 60/40/0/0 %
#   bei Breite 1/2/3/4 (5 Instanzen; bei 15 % Hindernissen 20 % bei Breite 1-3). Die Wurzel-Überraschung "mehr
#   Hindernisse, kleinere Lücke" ist hier also kein Trost: der Strahl verliert dort das Ziel statt der Qualität.
# RASTERGRÖSSE (Breite 3, 6/10/14/18/22): Lücke 0.0/0.0/0.0/1.0/5.8 % (Optimal-Anteil 80/80/60/40/0 %), Expansions-
#   Verhältnis 1.14/0.84/0.68/0.54/0.40 - eine feste Breite reicht bei größeren Instanzen nicht mehr, der Effizienz-
#   vorsprung gegen A* wächst dabei.
# HANDGEBAUTE FALLE: Breite 1 (wie GBFS) 7.47 % Lücke, Breite 2 findet den optimalen Pfad; 5 Expansionen gegen 7 bei A*.
# GRENZE DER SCHICHT-SUCHE: bei unbegrenzter Breite liefert Beam Search den Pfad mit den wenigsten KANTEN. Auf diesen
#   Rastern war er in allen getesteten Instanzen trotzdem optimal (Test); ein konstruiertes Gegenbeispiel (Direktkante
#   10 gegen Umweg 3) zeigt, dass es allgemein nicht gilt.

PRESETS = {
    "Standardfall (Voreinstellung)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "width": 3, "rank": "h"},
    "Handgebaute Falle (Breite 1)": {"network": "trap", "side": 12, "obstacle_pct": 15, "seed": 35, "width": 1, "rank": "h"},
    "Ziel verloren (Breite 1)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 100000, "width": 1, "rank": "h"},
    "Breiter scheitert (Breite 2)": {"network": "grid", "side": 12, "obstacle_pct": 40, "seed": 200007, "width": 2, "rank": "h"},
    "Breiter ist teurer (Breite 3)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 200001, "width": 3, "rank": "h"},
    "Breite 8 (teurer als A*)": {"network": "grid", "side": 12, "obstacle_pct": 15, "seed": 35, "width": 8, "rank": "h"},
    "Großes Raster (Breite 3)": {"network": "grid", "side": 22, "obstacle_pct": 15, "seed": 35, "width": 3, "rank": "h"},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Rastergröße 12, 15 % Hindernisse, Breite 3, Rangfolge h: im Median über die 5 festen Instanzen 2.8 % Lücke und 0.64x so viele Expansionen wie A\\* - aber 1 von 5 Läufen scheitert, nur 1 von 5 ist optimal. Diese Instanz (Seed 35) ist zufällig optimal.",
    "Handgebaute Falle (Breite 1)": "Der 8-Knoten-Graph aus der Wurzel: Breite 1 (ein Kandidat je Schicht, kein Zurück) fällt wie Greedy Best-First auf den Köder herein (7.47 % zu lang). Breite 2 findet den optimalen Pfad.",
    "Ziel verloren (Breite 1)": "Mit Breite 1 gibt es kein Zurück: in dieser Instanz (Seed 100000) läuft der einzige Kandidat in eine Sackgasse, die Suche scheitert und behauptet keinen Pfad. Greedy Best-First mit Backtracking findet hier einen.",
    "Breiter scheitert (Breite 2)": "40 % Hindernisse, Seed 200007: Breite 1 findet das Ziel, Breite 2 verliert es - erst ab Breite 3 klappt es wieder. Ein breiterer Strahl ist nicht automatisch besser (Nicht-Monotonie).",
    "Breiter ist teurer (Breite 3)": "Seed 200001: Breite 2 findet einen Pfad mit 190.3 km, Breite 3 einen längeren mit 193.6 km - ein breiterer Strahl mit höheren Kosten.",
    "Breite 8 (teurer als A*)": "Ab Breite 8 waren alle 5 festen Instanzen optimal - aber hier expandiert der Strahl 118 Knoten, A\\* nur 82. Zuverlässig optimal ist teurer als A\\*.",
    "Großes Raster (Breite 3)": "Rastergröße 22: dieselbe Breite 3 reicht nicht mehr (Lücke im Median 5.8 %, keine der 5 Instanzen optimal), spart aber 0.40x der Expansionen von A\\*.",
}
# Beobachtete Spannweite des MEDIANS des Expansions-Verhältnisses (Beam / A*) über die 5 festen Sweep-Instanzen
# (mit Sicherheitsabstand). Vollständig deterministisch. Die Falle ist ein fester Graph (5 / 7 = 0.71).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (0.5, 1.0),
    "Handgebaute Falle (Breite 1)": (0.65, 0.8),
    "Ziel verloren (Breite 1)": (0.15, 0.4),
    "Breiter scheitert (Breite 2)": (0.6, 1.0),
    "Breiter ist teurer (Breite 3)": (0.5, 1.0),
    "Breite 8 (teurer als A*)": (1.05, 1.5),
    "Großes Raster (Breite 3)": (0.3, 0.5),
}
