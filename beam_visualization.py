"""Plotly-Abbildungen: Rasterkarte, Strahl je Schicht (Strahl / verworfene Kandidaten / bereits expandiert),
Pfad-Überlagerung (Strahl, Greedy Best-First, A*), Sweeps über die Breite, Optimal-/Scheiter-Anteile, Kosten über die
Breite für EINE Instanz (Nicht-Monotonie). Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

NODE_COLOR = "#4c78a8"
BLOCKED_COLOR = "#9d755d"
ASTAR_COLOR = "#4c78a8"
BEAM_COLOR = "#e45756"
GBFS_COLOR = "#f58518"
GREEN = "#54a24b"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=460):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _start_goal_trace(inst):
    xy = inst.graph.xy
    return [
        go.Scatter(x=[xy[inst.start, 0]], y=[xy[inst.start, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#2ca02c", line=dict(width=1, color="white")), name="Start"),
        go.Scatter(x=[xy[inst.goal, 0]], y=[xy[inst.goal, 1]], mode="markers", marker=dict(size=16, symbol="star", color="#d62728", line=dict(width=1, color="white")), name="Ziel"),
    ]


def _blocked_trace(inst):
    return go.Scatter(x=inst.blocked_xy[:, 0], y=inst.blocked_xy[:, 1], mode="markers", marker=dict(size=6, symbol="square", color=BLOCKED_COLOR), name="Hindernis")


def build_instance(inst):
    xy = inst.graph.xy
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=6, color=NODE_COLOR, line=dict(width=1, color="white")), name="Offene Zellen"))
    if len(inst.blocked_xy):
        fig.add_trace(_blocked_trace(inst))
    fig.add_traces(_start_goal_trace(inst))
    return _map_layout(fig)


def build_layer_map(inst, per_layer, layer):
    """Zustand in Schicht `layer` (0-basiert): grau = in früheren Schichten expandiert, rot = Strahl dieser Schicht (wird
    jetzt expandiert), orange x = Kandidaten, die beim Beschneiden für die NÄCHSTE Schicht herausfielen (für immer verloren)."""
    xy = inst.graph.xy
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=5, color="rgba(76,120,168,0.25)"), name="Noch nicht berührt", hoverinfo="skip"))
    if len(inst.blocked_xy):
        fig.add_trace(_blocked_trace(inst))
    earlier = sorted({n for beam, _d in per_layer[:layer] for n in beam})
    if earlier:
        fig.add_trace(go.Scatter(x=xy[earlier, 0], y=xy[earlier, 1], mode="markers", marker=dict(size=8, color="rgba(120,120,120,0.55)"), name="Früher expandiert", hoverinfo="skip"))
    beam, dropped = per_layer[layer]
    if dropped:
        fig.add_trace(go.Scatter(x=xy[dropped, 0], y=xy[dropped, 1], mode="markers", marker=dict(size=10, symbol="x", color=GBFS_COLOR, line=dict(width=2, color=GBFS_COLOR)), name="Verworfen (verloren)"))
    fig.add_trace(go.Scatter(x=xy[beam, 0], y=xy[beam, 1], mode="markers", marker=dict(size=12, color=BEAM_COLOR, line=dict(width=1, color="white")), name="Strahl dieser Schicht"))
    fig.add_traces(_start_goal_trace(inst))
    return _map_layout(fig)


def build_paths(inst, beam_path, gbfs_path, astar_path):
    xy = inst.graph.xy
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers", marker=dict(size=5, color="rgba(76,120,168,0.35)"), name="Zellen", hoverinfo="skip"))
    if len(inst.blocked_xy):
        fig.add_trace(_blocked_trace(inst))
    if astar_path:
        p = xy[astar_path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color="rgba(76,120,168,0.55)", width=9), name="A* / UCS (optimal)"))
    if gbfs_path:
        p = xy[gbfs_path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", line=dict(color=GBFS_COLOR, width=2.5, dash="dash"), name="Greedy Best-First"))
    if beam_path:
        p = xy[beam_path]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines+markers", line=dict(color=BEAM_COLOR, width=3, dash="dot"), marker=dict(size=5, color=BEAM_COLOR), name="Beam Search"))
    fig.add_traces(_start_goal_trace(inst))
    return _map_layout(fig)


def build_sweep(rows, param_label, key, y_label, color=BEAM_COLOR, ref_line=None, ref_label=None):
    """Median als Linie, Minimum bis Maximum über die Instanzen als Band (`<key>_lo`/`<key>_hi`). `ref_line`: waagerechte
    Referenz (z. B. 1.0 = so viele Expansionen wie A*)."""
    xs = [r["value"] for r in rows]
    ys = [r[key] for r in rows]
    lo = [r[f"{key}_lo"] for r in rows]
    hi = [r[f"{key}_hi"] for r in rows]
    rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=y_label))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#888", dash="dash", width=1.5), annotation_text=ref_label, annotation_position="top left")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 360, legend_y=-0.3)


def build_share_bars(rows, param_label):
    """Anteil ALLER Läufe: optimal (Lücke 0) und gescheitert - gescheiterte zählen nie als gelöst."""
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["optimal_share"] for r in rows], name="optimal (Lücke 0)", marker_color=GREEN))
    fig.add_trace(go.Bar(x=xs, y=[r["failed_share"] for r in rows], name="gescheitert (kein Pfad)", marker_color=BLOCKED_COLOR))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text="Anteil der Läufe (%)", range=[0, 100])
    return _base(fig, 320, legend_y=-0.35)


def build_cost_by_width(widths, costs, optimum):
    """Pfadkosten in Abhängigkeit von der Strahlbreite für EINE Instanz; gescheiterte Läufe als x oberhalb der Kurve.
    Gestrichelt: das Optimum (Uniform-Cost). Wo die Kurve steigt, ist ein breiterer Strahl schlechter."""
    finite = [c for c in costs if c != float("inf")]
    top = (max(finite) if finite else optimum) * 1.06
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[w for w, c in zip(widths, costs) if c != float("inf")], y=finite, mode="lines+markers", line=dict(color=BEAM_COLOR, width=2.5), name="Pfadkosten Beam Search"))
    failed = [w for w, c in zip(widths, costs) if c == float("inf")]
    if failed:
        fig.add_trace(go.Scatter(x=failed, y=[top] * len(failed), mode="markers", marker=dict(size=11, symbol="x", color=BLOCKED_COLOR, line=dict(width=2, color=BLOCKED_COLOR)), name="gescheitert"))
    fig.add_hline(y=optimum, line=dict(color=ASTAR_COLOR, dash="dash", width=1.5), annotation_text="Optimum", annotation_position="bottom right")
    fig.update_xaxes(title_text="Strahlbreite k", dtick=1)
    fig.update_yaxes(title_text="Pfadkosten (km)")
    return _base(fig, 340, legend_y=-0.3)
