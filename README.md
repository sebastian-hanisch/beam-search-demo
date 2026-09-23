# Beam Search – wie viele Kandidaten reichen, wenn es kein Zurück gibt? – Streamlit-Demo

Viertes Stück der **Heuristische-Baumsuche-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" - ein Kind der Wurzel [greedy-best-first-demo](../greedy-best-first-demo): **Greedy Best-First Search (GBFS)** verfolgt **einen** Kandidaten und ist deshalb kurzsichtig (Lücke zum Optimum). **Beam Search** hält je Schicht die **k besten** Kandidaten parallel und beschneidet den Rest - aber **ohne Zurück**: was aus dem Strahl fällt, ist unwiederbringlich verloren. Die Breite k ist ein Regler zwischen "ein Pfad" (k = 1) und "fast alles".

**Einordnung in die Linie:** derselbe Graph, dieselbe Instanz, derselbe Suchkern wie in [astar-demo](../astar-demo) und [ida-star-demo](../ida-star-demo) (dort korrektheitsgeprüft); GBFS, A\* und Uniform-Cost dienen als Vergleichsgrößen. Neu ist `beam_search` mit der Rangfolge **h** (nur Heuristik, Kind der Wurzel) oder **f = g + h** (wie A\*).

```
Greedy Best-First Search (Wurzel)                                                          [gebaut]
 ├─ Beam Search → {Diverse Beam Search, Monobeam}          [Beam Search = DIESES STÜCK, Fortsetzungen nicht gebaut]
 ├─ A* → Iterative Deepening A* (IDA*)                                                     [gebaut]
 └─ Monte Carlo Tree Search (MCTS)                                                         [nicht gebaut]
Beam Search + A* → Beam Stack Search (Konvergenzpunkt)                                     [nicht gebaut]
```

Ergebnis in Kürze: Die Vorab-Hypothese "**ein breiterer Strahl macht das Ergebnis besser - monoton**" gilt **im Median, aber nicht je Instanz**: die Lücke sinkt von 16 % (Breite 1) auf 0 % (ab Breite 5), ab Breite 8 waren alle 5 Instanzen optimal - doch bei 2 bis 14 % der Instanzen ist irgendein breiterer Strahl **schlechter** (höhere Kosten, oder er **scheitert**, obwohl ein schmalerer das Ziel fand). Der eigentliche Preis eines schmalen Strahls ist nicht die Lücke, sondern das **Scheitern** (kein Zurück): 20 % der Läufe bei Breite 1-3, bei 40 % Hindernissen 60 % (Breite 1). Und billiger als A\* ist der Strahl **nur solange man Lücke oder Scheitern in Kauf nimmt**: bis Breite 5 weniger Expansionen als A\*, ab Breite 6 mehr - zuverlässig optimal (Breite 8) kostet er das 1.24-Fache von A\*.

| Frage | Ergebnis (Rastergröße 12, Hindernisdichte 15 %, Rangfolge h, sofern nicht anders angegeben; **Median** über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch; Lücke und Verhältnisse nur über gelöste Läufe) |
|---|---|
| **Macht mehr Breite es besser?** | ✅ im Median: Lücke **16.2/7.3/2.8/0.35/0.0/0.0/0.0 %** bei Breite 1/2/3/4/5/6/8; Optimal-Anteil (aller Läufe) 0/0/20/40/60/60/100 % - GBFS (Wurzel) hat 14.7 % Lücke |
| **Ist es je Instanz monoton?** | ⚠️ NEIN: über 50 feste Instanzen (Breiten 1–24) ist bei **6 %** (Größe 12, 15 % Hindernisse), **2 %** (0 %), **10 %** (40 %) und **14 %** (Größe 20) irgendein breiterer Strahl schlechter - bei 40 % Hindernissen ausschließlich als **Scheitern** trotz Erfolg eines schmaleren (Beispiel Seed 200007: Breite 1 findet das Ziel, Breite 2 scheitert, ab Breite 3 klappt es wieder) |
| **Scheitert ein schmaler Strahl?** | ⚠️ ja: Scheiter-Quote **20 %** bei Breite 1–3 (15 % Hindernisse), danach 0 %; bei 40 % Hindernissen **60/40/0/0 %** bei Breite 1/2/3/4. GBFS mit Backtracking scheitert in diesen Läufen nie |
| **Ist er billiger als A\*?** | ⚠️ nur mit Lücke/Scheitern: Expansions-Verhältnis Beam / A\* **0.27/0.47/0.64/0.78/0.97/1.08/1.24** bei Breite 1/2/3/4/5/6/8; gesättigt bei ~1.25 (Uniform-Cost 121, A\* 93, breiter Strahl 120 Expansionen im Median). Gespeicherte Knoten 0.45/0.60/0.71/0.85/0.94/1.00/1.10 |
| **Hilft die Rangfolge f = g + h?** | teils: bei Breite 2–4 kleinere Lücke (4.5/0.35/0.0 gegen 7.3/2.8/0.35 %) und bei Breite 2 keine Ausfälle (0 gegen 20 %) - aber bei Breite 1 **mehr** Ausfälle (40 gegen 20 %, nur 5 Instanzen); Nicht-Monotonie bleibt |
| **Hindernisdichte-Sweep (Breite 3; 0/10/20/30/40 %)** | Lücke 2.7/0.0/1.3/0.0/0.0 %, Scheiter-Quote 0/0/20/0/0 %, Expansions-Verhältnis **0.61/0.72/0.82/0.98/1.00** - der Vorteil gegen A\* schmilzt mit den Hindernissen |
| **Rastergrößen-Sweep (Breite 3; 6/10/14/18/22)** | Lücke 0.0/0.0/0.0/1.0/5.8 %, Optimal-Anteil 80/80/60/40/0 %, Expansions-Verhältnis 1.14/0.84/0.68/0.54/0.40 - eine feste Breite reicht bei größeren Instanzen nicht mehr |
| **Handgebaute Heuristik-Falle** | Breite 1 fällt wie GBFS auf den Köder herein (7.47 % zu lang, 5 Expansionen gegen 7 bei A\*); Breite 2 findet den optimalen Pfad |

## Was die Demo zeigt

1. **Beam Search in Aktion** (Schritt-Slider): **Instanz** → **Strahl je Schicht** (Schicht-Slider: grau = früher expandiert, rot = Strahl der Schicht, orange x = beim Beschneiden verworfen, für immer verloren; bei Scheitern endet die Suche sichtbar in der Sackgasse) → **Ergebnis** (Beam, Greedy Best-First und A\* überlagert; bei Scheitern ohne Pfadbehauptung).
2. **Was der Strahl gegen A\* bringt:** Lücke zum Optimum, Expansionen / A\*, Speicher / A\*, Status.
3. **📐 Sweeps** über Breite, Hindernisdichte und Rastergröße (Median, Min-Max-Band, Referenzlinie "so viel wie A\*"), wählbare Kennzahl (Lücke / Expansionen / Speicher / Optimal- und Scheiter-Anteil), 5 feste Instanzen ab Seed 100000.
4. **🔬 Experiment:** "Wird es mit mehr Breite immer besser?" - Pfadkosten dieser einen Instanz für Breite 1–24 (Scheitern als x), plus die Nicht-Monotonie-Rate über 50 feste Instanzen.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Instanz (Raster / handgebaute Heuristik-Falle), Rastergröße (5–25), Hindernisdichte (0–40 %), Seed der Instanz (+ 🎲), **Strahlbreite** (1 bis 24), **Rangfolge** (h / f = g + h). Kein Zufall im Kern, kein Ketten-Seed, kein Bewertungsbudget - vollständig deterministisch.

## Messwerte der Presets (einzelne Instanz)

| Preset | Instanz | Ergebnis dieser Instanz |
|---|---|---|
| Standardfall (Voreinstellung) | Seed 35, Breite 3 | zufällig optimal: 63 Expansionen gegen 82 bei A\*, 84 gegen 100 gespeicherte Knoten (im Median über die 5 festen Instanzen: 2.8 % Lücke, 1 von 5 gescheitert) |
| Handgebaute Falle (Breite 1) | Falle, Breite 1 | 7.47 % zu lang wie GBFS, 5 gegen 7 Expansionen |
| Ziel verloren (Breite 1) | Seed 100000, Breite 1 | gescheitert nach 13 Schichten (Sackgasse), GBFS mit Backtracking findet einen Pfad |
| Breiter scheitert (Breite 2) | Seed 200007, 40 % Hindernisse | Breite 1 findet das Ziel, Breite 2 scheitert, ab Breite 3 klappt es wieder |
| Breiter ist teurer (Breite 3) | Seed 200001 | Breite 2: 190.3 km, Breite 3: 193.6 km |
| Breite 8 (teurer als A\*) | Seed 35, Breite 8 | optimal, aber 118 Expansionen gegen 82 bei A\* |
| Großes Raster (Breite 3) | Größe 22, Seed 35 | 5.8 % Lücke, 122 gegen 317 Expansionen bei A\* |

Die einzelne Instanz weicht von den Sweep-Medianen ab - die Mediane oben sind die belastbaren Zahlen; jedes Preset prüft sich zusätzlich über die 5 festen Sweep-Instanzen gegen eine gemessene Spannweite (siehe `tests/test_presets.py`), und die Aussagen der Presets (gescheitert, breiter scheitert, breiter ist teurer, Falle) sind als eigene Tests hinterlegt.

## Modell und Verfahren

- **Instanz und Graph** (`beam_scenario.py`, `beam_graph.py`): wortgleiche Kopien aus [astar-demo](../astar-demo) - gestörtes Raster mit Hindernissen, Kantengewicht = echter euklidischer Abstand, plus die handgebaute 8-Knoten-Falle.
- **Suchkern** (`beam_algorithm.py`): der `_search`-Kern samt `greedy_best_first`/`uniform_cost_search`/`a_star` aus der A\*-Demo. NEU `beam_search`: schichtweise (Schicht = Kantenzahl vom Start); je Schicht werden alle Strahl-Knoten expandiert, die Nachfolger gesammelt (je Knoten das kleinste g = Duplikaterkennung innerhalb der Schicht; schon expandierte Knoten früherer Schichten kommen nicht wieder auf), nach der Rangfolge (`h` oder `g + h`, Tie-Break Knotenindex) auf die besten k beschnitten. Wird das Ziel erzeugt, endet die Suche mit dem kleinsten g unter den Zielkandidaten dieser Schicht; eine leere Kandidatenmenge heißt `failed` - **kein Pfad wird behauptet**.
- **Auswertung** (`beam_evaluation.py`): Lücke ggü. Uniform-Cost **nur über gelöste Läufe**, daneben stets **Scheiter-Quote** und **Optimal-Anteil** (Anteil ALLER Läufe mit Lücke 0, gescheiterte zählen als nicht optimal - ein Median nur über die Überlebenden täuscht sonst bei schmalem Strahl); Expansions- und Speicher-Verhältnis ggü. A\*; Sweeps mit Median und Min-Max-Spanne; `monotonicity` = Anteil der Instanzen, bei denen ein breiterer Strahl schlechter ist (höhere Kosten oder Scheitern trotz Erfolg eines schmaleren).

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "ein breiterer Strahl macht das Ergebnis besser - monoton" - nur im Mittel bestätigt, je Instanz WIDERLEGT.** 2–14 % der Instanzen werden mit mehr Breite schlechter. Das ist genau die Schwäche, an der **Monobeam** (Lemons et al. 2022) ansetzt - hier nur gemessen, nicht behoben.
- **Kein Zurück = Scheitern.** Ein schmaler Strahl verliert das Ziel in einer Sackgasse; bei dichten Hindernissen häufig (40 %: 60 % bei Breite 1). Die Wurzel-Überraschung "mehr Hindernisse, kleinere Lücke" ist hier kein Trost: der Strahl verliert dort das Ziel statt der Qualität. Beam Search + Backtracking ist **Beam Stack Search** (Zhou & Hansen 2005) - nicht gebaut.
- **Billiger als A\* nur mit Verzicht.** Zuverlässig optimal (5 von 5) ist der Strahl erst bei Breite 8 - dort teurer als A\* (1.24x Expansionen). Die "kleinste Breite, die optimal ist" ist außerdem nur im Nachhinein bekannt.
- **Der gezählte Speicher ist der von A\*-Art:** alle entdeckten Knoten (mit Closed-Set zur Duplikaterkennung). Ein echter Speicher-Strahl ohne Closed-Set bräuchte nur die Spitzenbreite (Median 6 bei Breite 3) - diese Variante ist nicht gebaut.
- **Unbegrenzte Breite findet die wenigsten KANTEN, nicht zwingend den billigsten Pfad.** Auf diesen Rastern war das Ergebnis in allen getesteten Instanzen (120) trotzdem optimal; ein konstruiertes Gegenbeispiel (Direktkante 10 gegen Umweg 3) zeigt, dass es allgemein nicht gilt (als Test hinterlegt).
- **Ursachen nicht isoliert:** warum die Rangfolge f bei Breite 1 mehr Ausfälle hat, oder warum die Nicht-Monotonie bei Größe 20 häufiger ist als bei Größe 12, wird hier nicht getrennt untersucht - gemessen ist nur das Ergebnis (5 bzw. 50 Instanzen).
- **Nicht gebaut:** Diverse Beam Search, Monobeam, Beam Stack Search, Stochastic Beam Search.
- **Synthetische Instanzen:** ein Raster mit Jitter, Vierer-Nachbarschaft, keine Zeitfenster, keine gerichteten Kanten. Andere Graphstrukturen wurden nicht gemessen.

## Verifikation

- **Gültiger Pfad** (zusammenhängend, Start bis Ziel, ohne Wiederholung, Kosten gegen unabhängige Neuberechnung) für beide Rangfolgen und Breiten 1–8; **Kosten nie unter dem Optimum** (gegen vollständige Enumeration aller einfachen Pfade auf kleinen Instanzen); Determinismus.
- **Drei Sonderfälle direkt nachgewiesen:** ein Kettengraph wird von jeder Breite mit einer Expansion je Knoten gelöst; bei Einheitsgewichten, h ≡ 0 und unbegrenzter Breite ist Beam Search Breitensuche (Kosten == Uniform-Cost-Optimum == Brute-Force); Breite 1 mit Rangfolge h ist ein gieriger Abstieg, gegen eine **unabhängig geschriebene Referenzimplementierung** über 40 Instanzen identisch (Pfad und Scheitern).
- **Schicht- und Duplikat-Buchführung:** je Schicht höchstens k Strahl-Knoten, kein Knoten zweimal expandiert, Summe der Strahlgrößen == Expansionen, Pfadlänge == Schichtzahl, verworfene Kandidaten nie im Strahl.
- **Scheitern:** ein nicht erreichbares Ziel scheitert immer und behauptet keinen Pfad; ein schmaler Strahl scheitert auf echten Instanzen, wo ein breiterer das Ziel findet.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt**, über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/`ev.monotonicity`), NIE über ein Ad-hoc-Skript; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, jede Schicht, beide Instanz-Typen, gescheiterte Läufe in jedem Schritt, Extremwerte, Würfel, Permalink-Grenzen inkl. Breite/Rangfolge, Instanzwechsel, ausgeblendete Regler bei der Falle, Sweeps und Experiment auf Abruf, Footer). 464 Tests.

Literatur: Lowerre, B. T. (1976). *The HARPY Speech Recognition System.* Dissertation, Carnegie Mellon University (erste Verwendung von Beam Search). Zhou, R., & Hansen, E. A. (2005). *Beam-Stack Search: Integrating Backtracking with Beam Search.* Proceedings of the 15th International Conference on Automated Planning and Scheduling (ICAPS). Lemons, S., Linares López, C., Holte, R. C., & Ruml, W. (2022). *Beam Search: Faster and Monotonic.* Proceedings of the International Conference on Automated Planning and Scheduling, 32(1), 222-230.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Instanz-Umschalter, Schritte (mit Schicht-Slider), Ergebnis, 📐 Sweeps, 🔬 Experiment, 🚧 Grenzen, Mathe |
| `beam_algorithm.py` | Gemeinsamer Suchkern (Kopie aus der A\*-Demo) + `beam_search` |
| `beam_graph.py`, `beam_scenario.py` | Graph, Raster- und Fallen-Instanz (Kopie) |
| `beam_constants.py` | Konstanten, Presets, gemessene Werte |
| `beam_evaluation.py` | Lücke, Scheiter-Quote, Verhältnisse, Sweeps, Nicht-Monotonie |
| `beam_presets.py`, `beam_visualization.py` | Permalink/Presets, Plotly-Figuren (Schicht-Karte, Pfad-Überlagerung, Sweeps, Kosten über die Breite) |
| `tests/` | Zentrale Korrektheitskette, Szenario/Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
