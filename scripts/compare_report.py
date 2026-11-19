"""Informe HTML: trayectorias estimadas (tracker) frente a reales (GT), por iteración.

Uso (desde la raíz del repo, tras `etl` y `track`):
    python scripts/compare_report.py
    python scripts/compare_report.py --tracks-b otra/tracks.parquet   # comparar dos runs
    python scripts/compare_report.py --out reports/compare.html

Usa los mismos cálculos que el visor (`src/compare.py`) y las mismas gráficas
(`visualization/tracks_layer.py`). Solo lee; no toca el tracker.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "visualization"))

from src.compare import compare_runs, run_compare  # noqa: E402
from src.config import load_config  # noqa: E402
import tracks_layer as TL  # noqa: E402

BG, PANEL, TEXT, MUTED = "#0f1419", "#161d24", "#d9e1e8", "#8a97a3"
TEAM = {"BLUE": ["#2f6bff", "#4fb6ff", "#7a8cff", "#1e90ff"], "RED": ["#ff2a2a", "#ff8a00", "#ff6f61", "#ffb347", "#c00000"]}
PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"


def drone_colors(gt_view: dict) -> dict:
    out, idx = {}, {}
    for view in sorted(set(gt_view.values())):
        team = view.split(" · ")[0]
        pal = TEAM.get(team, ["#cccccc"])
        out[view] = pal[idx.get(team, 0) % len(pal)]
        idx[team] = idx.get(team, 0) + 1
    return out


def top_view(gt: pd.DataFrame, tracks: pd.DataFrame, run: TL.Run, gt_view: dict, colors: dict) -> go.Figure:
    """Vista cenital en ENU del ETL: GT continuo, tracks discontinuos."""
    fig = go.Figure()
    for tid, g in tracks.sort_values("ts_ms").groupby("track_id"):
        g = g.iloc[::10]  # 1 punto/s
        fig.add_trace(go.Scatter(x=g.e, y=g.n, mode="lines", line=dict(color=run.colors.get(tid, "#ffe066"), width=1.5,
                                                                          dash="dash"),
                                 opacity=0.85, showlegend=False, hovertemplate=f"Track …{tid[-6:]}<extra></extra>"))
    for key, g in gt.sort_values("ts_ms").groupby("drone_key"):
        view = gt_view.get(key, key)
        fig.add_trace(go.Scatter(x=g.e, y=g.n, mode="lines", line=dict(color=colors.get(view, "#ccc"), width=3),
                                 name=view, hovertemplate=f"{view}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="#ffe066", dash="dash"),
                             name="Tracks estimados"))
    axis = dict(gridcolor="#2a3540", zerolinecolor="#3a4855", color=MUTED)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, size=11), height=460,
                      margin=dict(l=50, r=10, t=10, b=40), legend=dict(font=dict(size=10)),
                      xaxis=dict(axis, title="Este (m)"), yaxis=dict(axis, title="Norte (m)", scaleanchor="x"))
    return fig


def table(df: pd.DataFrame, delta_cols: dict | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            txt = "–" if (isinstance(v, float) and pd.isna(v)) else (f"{v:.1f}" if isinstance(v, float) else html.escape(str(v)))
            cls = ""
            if delta_cols and c in delta_cols and isinstance(v, (int, float)) and pd.notna(v) and v != 0:
                cls = ' class="good"' if (v < 0) == (delta_cols[c] == "low") else ' class="bad"'
            cells.append(f"<td{cls}>{txt}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


ITER_COLS = {"gospa_mean": "GOSPA medio (m)", "gospa_loc": "· posición", "gospa_missed": "· perdidos",
             "gospa_false": "· falsos", "pct_tracked": "% seguido", "err_mean_m": "Error medio (m)",
             "id_switches": "Cambios de track", "tracks_never_matched": "Tracks falsos"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--tracks", default=None, help="por defecto reports/tracks.parquet")
    ap.add_argument("--tracks-b", default=None)
    ap.add_argument("--out", default=None, help="por defecto reports/compare_tracks.html")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tracks_a = pd.read_parquet(a.tracks or cfg.reports / "tracks.parquet")
    tracks_b = pd.read_parquet(a.tracks_b) if a.tracks_b else None
    gt_all = pd.read_parquet(cfg.interim / "ground_truth.parquet")
    groups = sorted(gt_all.split_group.unique())

    sections, it_rows = [], {"A": [], "B": []}
    nav = []
    for group in groups:
        gt = gt_all[gt_all.split_group == group]
        gt_view = {k: TL.viewer_drone_key(k) for k in gt.drone_key.unique()}
        colors = drone_colors(gt_view)
        runs = {}
        for name, tr in (("A", tracks_a), ("B", tracks_b)):
            if tr is None:
                continue
            t = tr[tr.split_group == group].copy()
            t["track_id"] = t.track_id.astype(str)
            cmp = run_compare(t, gt, cfg)
            ids = sorted(t.track_id.unique())
            runs[name] = (t, TL.Run(name=name, tracks=t, compare=cmp,
                                    colors={tid: TL.TRACK_COLORS[i % len(TL.TRACK_COLORS)] for i, tid in enumerate(ids)}))
            it_rows[name].append(cmp["iterations"])
        layer = TL.TrackLayer(split_group=group, runs={k: v[1] for k, v in runs.items()}, gt_has=True,
                              labelled=pd.DataFrame(), gt_view=gt_view)
        anchor = group.replace(":", "-")
        nav.append(f'<a href="#{anchor}">{html.escape(group.replace("test:", ""))}</a>')
        body = [f'<section id="{anchor}"><h2>{html.escape(group)}</h2>']
        for name, (t, run) in runs.items():
            it = run.compare["iterations"].iloc[0]
            body.append(f'<p class="kpi"><b>Run {name}</b> · GOSPA {it.gospa_mean:.1f} m · {it.pct_tracked:.0f}% del tiempo '
                        f'seguido · {int(it.id_switches)} cambios de track · {int(it.tracks_never_matched)} tracks falsos</p>')
            top = top_view(gt, t, run, gt_view, colors).to_html(full_html=False, include_plotlyjs=False)
            cmp_fig = TL.comparison_figure(go, make_subplots, layer, name, None, None, colors, BG, TEXT, MUTED)
            cmp_fig.update_layout(height=430)
            d = run.compare["drones"].copy()
            d.insert(0, "Dron", d.drone_key.map(gt_view))
            d = d[["Dron", "pct_tracked", "err_mean_m", "err_p90_m", "n_tracks", "id_switches", "first_track_delay_s"]]
            d.columns = ["Dron", "% seguido", "Error medio (m)", "Error p90 (m)", "Tracks usados", "Cambios de track",
                         "Retraso 1er track (s)"]
            body.append(f'<div class="grid"><div class="plot">{top}</div><div class="plot">'
                        f'{cmp_fig.to_html(full_html=False, include_plotlyjs=False)}</div></div>{table(d)}')
        body.append("</section>")
        sections.append("".join(body))

    def iter_table(name):
        df = pd.concat(it_rows[name]) if it_rows[name] else pd.DataFrame()
        return df[["split_group"] + list(ITER_COLS)].rename(columns={"split_group": "Iteración", **ITER_COLS})

    summary = table(iter_table("A"))
    if tracks_b is not None:
        cmp = compare_runs(pd.concat(it_rows["A"]), pd.concat(it_rows["B"]), "split_group")
        cols = {"gospa_mean": "low", "pct_tracked": "high", "err_mean_m": "low", "id_switches": "low",
                "tracks_never_matched": "low"}
        out = cmp[["split_group"] + [f"{c}_diff" for c in cols]]
        out.columns = ["Iteración"] + [f"Δ {ITER_COLS[c]}" for c in cols]
        summary = ("<h3>Run A</h3>" + summary + "<h3>Run B</h3>" + table(iter_table("B")) +
                   "<h3>Diferencia B − A (verde = B mejora)</h3>" +
                   table(out, {f"Δ {ITER_COLS[c]}": v for c, v in cols.items()}))

    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracker frente a realidad</title><script src="{PLOTLY}"></script>
<style>
:root{{color-scheme:dark}}
body{{background:{BG};color:{TEXT};font:14px/1.5 "Segoe UI",system-ui,sans-serif;margin:0;padding:24px max(16px,env(safe-area-inset-left,0px))}}
.wrap{{max-width:1300px;margin:0 auto}}
h1{{margin:0 0 4px;font-size:28px}} h2{{margin:32px 0 6px;font-size:20px;border-top:1px solid #24303a;padding-top:18px}}
h3{{font-size:14px;color:{MUTED};margin:14px 0 4px}}
p.lead{{color:{MUTED};max-width:85ch}} p.kpi{{margin:4px 0}}
nav a{{color:#7fb3ff;margin-right:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:12px}}
.plot{{min-width:0;background:{BG}}}
.tw{{overflow-x:auto;margin:8px 0 6px}}
table{{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 10px;border-bottom:1px solid #24303a;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:{MUTED};font-weight:600}}
td.good{{color:#5ee36a;font-weight:600}} td.bad{{color:#ff6f61;font-weight:600}}
ul.help{{color:{MUTED};font-size:13px;max-width:90ch}}
</style></head><body><div class="wrap">
<h1>Tracker frente a realidad</h1>
<p class="lead">Trayectorias estimadas por el tracker comparadas con las reales (Dronetag), iteración a iteración.
Cada segundo se empareja cada track con el dron real más cercano, con el mismo criterio que GOSPA (corte {float(cfg.data.get('evaluation', {}).get('gospa_c_m', 50)):.0f} m).</p>
<nav>{''.join(nav)}</nav>
<h2>Resumen</h2>{summary}
<ul class="help">
<li><b>GOSPA</b>: error global en metros (más bajo = mejor). Se reparte en error de <b>posición</b>, drones <b>perdidos</b> y tracks <b>falsos</b>.</li>
<li><b>% seguido</b>: parte del tiempo en que el dron real tiene un track a menos del corte.</li>
<li><b>Cambios de track</b>: veces que el dron pasa a estar seguido por otro track (el tracker "se confunde").</li>
<li><b>Tracks falsos</b>: tracks que nunca coinciden con ningún dron real.</li>
<li>En la cobertura, cada color de barra es un track distinto; las marcas rojas son segundos sin track.</li>
</ul>
{''.join(sections)}
</div></body></html>"""
    out = Path(a.out) if a.out else cfg.reports / "compare_tracks.html"
    out.write_text(page, encoding="utf-8")
    print(f"Informe: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
