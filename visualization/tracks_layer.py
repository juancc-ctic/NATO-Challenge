"""Capa de trayectorias estimadas (tracker) y comparación con el GT para el visor.

Carga `reports/tracks.parquet` (uno o dos runs), el GT y las etiquetas del
interim del ETL, y lo deja todo en las coordenadas del visor:

- Los tracks están en ENU con el origen del ETL (`data/interim/origins.json`).
  El visor usa otro origen (mediana de sensores y drones). Para no mezclarlos,
  los tracks se pasan a lat/lon con su origen y se reproyectan con el del visor.
- La comparación (emparejamiento, errores, cobertura) se calcula con
  `src/compare.py` en ENU del ETL, igual que el evaluador.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare import add_geodetic, horizontal_ellipse, load_origins, run_compare  # noqa: E402

_DRONE_RE = re.compile(r"DroneTag_([A-Z]+)_\d{6}_(\d{4})_\d{4}_(.+?)(?:-\d+)?$")
TRACK_COLORS = ["#ffe066", "#7cf29a", "#c9a2ff", "#5fe1d3", "#ff9de2", "#d9f27c", "#ffc285", "#a0d8ff",
                "#f7f7f7", "#b5f5c8"]


def viewer_drone_key(drone_key: str) -> str:
    """`test:S1:I1:DroneTag_RED_240916_0731_0736_FPV` -> `RED · FPV 07:31` (clave de dron del visor).

    Debe coincidir con `data_loader._read_drone`.
    """
    stem = str(drone_key).split(":")[-1]
    m = _DRONE_RE.search(stem)
    return f"{m.group(1)} · {m.group(3)} {m.group(2)[:2]}:{m.group(2)[2:]}" if m else stem


def infer_split_group(iteration_dir: str) -> str | None:
    """Deduce `test:S1:I1` de una ruta `.../Testing datasets/Scenario 1/Iteration 1`."""
    path = str(iteration_dir).replace("\\", "/")
    sc = re.search(r"Scenario\s*(\d+)", path)
    it = re.search(r"Iteration\s*(\d+)", path)
    if not (sc and it):
        return None
    if "Testing" in path:
        return f"test:S{sc.group(1)}:I{it.group(1)}"
    ds = re.search(r"Dataset\s*([IVX]+)", path)
    return f"eval:Dataset_{ds.group(1)}:S{sc.group(1)}:I{it.group(1)}" if ds else None


@dataclass
class Run:
    name: str
    tracks: pd.DataFrame                    # 1 fila por (track, segundo) en coords del visor
    colors: dict = field(default_factory=dict)
    compare: dict = field(default_factory=dict)  # tablas de src.compare (vacío si no hay GT)


@dataclass
class TrackLayer:
    split_group: str
    runs: dict                              # nombre -> Run
    gt_has: bool
    labelled: pd.DataFrame                  # detecciones con posición y etiqueta, coords del visor
    gt_view: dict                           # drone_key -> clave de dron del visor


def _to_view(frame: pd.DataFrame, ds, t0: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy()
    out["ve"], out["vn"] = ds.proj.enu(out["lon"].to_numpy(), out["lat"].to_numpy())
    out["z_raw"] = out["alt_m"]
    out["z_agl"] = out["alt_m"] - ds.ground_ref
    out["ts"] = (pd.to_datetime(out["ts_ms"], unit="ms") - t0).dt.total_seconds()
    return out


def _load_run(name, path, group, origins, gt, ds, t0, cfg) -> Run:
    tracks = pd.read_parquet(path)
    tracks = tracks[tracks.split_group == group].copy()
    tracks["track_id"] = tracks.track_id.astype(str)
    compare = run_compare(tracks, gt, cfg) if len(gt) and len(tracks) else {}
    # Para dibujar basta 1 punto por segundo y track.
    tracks["sec"] = tracks.ts_ms // 1000
    draw = tracks.sort_values("ts_ms").drop_duplicates(["track_id", "sec"])
    draw = _to_view(add_geodetic(draw, origins), ds, t0)
    ids = sorted(draw.track_id.unique())
    colors = {tid: TRACK_COLORS[i % len(TRACK_COLORS)] for i, tid in enumerate(ids)}
    return Run(name=name, tracks=draw, colors=colors, compare=compare)


def load_layer(ds, t0, interim: str, split_group: str, tracks_paths: dict, cfg=None) -> TrackLayer:
    """Carga runs, comparación y detecciones etiquetadas de una iteración."""
    interim = Path(interim)
    origins = load_origins(interim / "origins.json")
    gt = pd.read_parquet(interim / "ground_truth.parquet")
    gt = gt[gt.split_group == split_group]
    runs = {name: _load_run(name, p, split_group, origins, gt, ds, t0, cfg) for name, p in tracks_paths.items() if p}

    labelled = pd.DataFrame()
    lab_path = interim / "detection_labels.parquet"
    if lab_path.exists():
        lab = pd.read_parquet(lab_path)
        lab = lab[lab.split_group == split_group]
        det = pd.read_parquet(interim / "detections.parquet",
                              columns=["row_uid", "lat", "lon", "alt_m", "has_position"])
        lab = lab.merge(det, on="row_uid", how="left")
        lab = lab[lab.has_position.fillna(False) & lab.lat.notna() & lab.lon.notna()]
        if len(lab):
            labelled = _to_view(lab, ds, t0)
            labelled["drone_view"] = labelled.label_drone_key.map(
                lambda k: viewer_drone_key(k) if isinstance(k, str) else None)
    gt_view = {k: viewer_drone_key(k) for k in gt.drone_key.unique()}
    return TrackLayer(split_group=split_group, runs=runs, gt_has=len(gt) > 0, labelled=labelled, gt_view=gt_view)


def track_traces(go, layer: TrackLayer, run_name: str, t_now: float, t_lo: float, zc: str, z_max: float,
                 opts: list, ds, drone_colors: dict, ell_scale: float = 1.0):
    """Trazas 3D: tracks estimados, líneas de error real↔estimado y elipses 2σ.

    `ell_scale` amplía la elipse solo para verla (radio 2σ típico ~10 m en escenas de ~3 km);
    la leyenda indica la ampliación para no confundirla con la incertidumbre real.
    """
    traces = []
    run = layer.runs.get(run_name)
    if run is None:
        return traces
    d = run.tracks[(run.tracks.ts <= t_now) & (run.tracks.ts >= t_lo)]
    if "tracks" in opts:
        for tid, g in d.groupby("track_id"):
            col = run.colors[tid]
            custom = np.stack([g.track_id.str[-6:], g.classification.fillna("Unknown").astype(str),
                               np.sqrt(np.clip(g.C_nn.to_numpy() + g.C_ee.to_numpy(), 0, None)).round(1)], -1)
            traces.append(go.Scatter3d(
                x=g.ve, y=g.vn, z=g[zc].fillna(0).clip(upper=z_max), mode="lines",
                line=dict(color=col, width=3, dash="dash"), name=f"Track …{tid[-6:]}", legendgroup="tracks",
                showlegend=False, customdata=custom,
                hovertemplate=("<b>Track estimado …%{customdata[0]}</b><br>Clase: %{customdata[1]}"
                               "<br>Incertidumbre horizontal (1σ): %{customdata[2]} m<extra></extra>")))
            last = g.iloc[-1]
            if t_now - last.ts <= 1.5:
                traces.append(go.Scatter3d(
                    x=[last.ve], y=[last.vn], z=[min(last[zc] if pd.notna(last[zc]) else 0, z_max)],
                    mode="markers", marker=dict(size=6, color=col, symbol="diamond-open", line=dict(width=2)),
                    showlegend=False, hoverinfo="skip"))
        traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines", name=f"Tracks estimados ({run_name})",
                                   line=dict(color="#ffe066", width=3, dash="dash"), legendgroup="tracks"))

    now = run.tracks[(run.tracks.ts <= t_now) & (run.tracks.ts > t_now - 1.5)]
    now = now.sort_values("ts").drop_duplicates("track_id", keep="last")

    if "ellipse" in opts and len(now):
        scale = float(ell_scale or 1.0)
        xs, ys, zs, txt = [], [], [], []
        for row in now.itertuples():
            pts = horizontal_ellipse(row.C_nn, row.C_ne, row.C_ee, k=2.0)
            if not len(pts):
                continue
            real = float(np.abs(pts).max())
            pts = pts * scale
            z = min(getattr(row, zc) if pd.notna(getattr(row, zc)) else 0, z_max)
            xs += list(row.ve + pts[:, 1]) + [None]
            ys += list(row.vn + pts[:, 0]) + [None]
            zs += [z] * len(pts) + [None]
            txt += [f"Track …{row.track_id[-6:]} · semieje mayor 2σ real: {real:.0f} m"] * len(pts) + [None]
        label = "Incertidumbre 2σ" + ("" if scale == 1 else f" (dibujada ×{scale:g})")
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#ffe066", width=3),
                                   opacity=0.8, name=label, text=txt, hovertemplate="%{text}<extra></extra>"))

    if "errors" in opts and run.compare:
        pairs = run.compare["pairs"]
        ts_ms_now = int((ds.t0 + pd.Timedelta(seconds=t_now)).value // 1_000_000)
        recent = pairs[(pairs.ts_ms <= ts_ms_now) & (pairs.ts_ms > ts_ms_now - 1500)]
        recent = recent.sort_values("ts_ms").drop_duplicates("drone_key", keep="last")
        xs, ys, zs, txt = [], [], [], []
        for p in recent.itertuples():
            dr = ds.drones.get(layer.gt_view.get(p.drone_key))
            tr = now[now.track_id == p.track_id]
            if dr is None or tr.empty:
                continue
            g = dr.df[dr.df.ts <= t_now]
            if g.empty:
                continue
            g = g.iloc[-1]
            t = tr.iloc[0]
            xs += [g.e, t.ve, None]
            ys += [g.n, t.vn, None]
            zs += [min(g[zc], z_max), min(t[zc] if pd.notna(t[zc]) else 0, z_max), None]
            txt += [f"{p.dist_m:.0f} m"] * 2 + [None]
        if xs:
            traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#ffffff", width=4),
                                       name="Error real↔estimado", text=txt,
                                       hovertemplate="Error: %{text}<extra></extra>"))
    return traces


def label_traces(go, layer: TrackLayer, t_now: float, t_lo: float, zc: str, z_max: float, drone_colors: dict):
    """Detecciones con posición coloreadas por el dron al que las asignó el etiquetado."""
    lab = layer.labelled
    if lab.empty:
        return []
    d = lab[(lab.ts <= t_now) & (lab.ts >= t_lo)]
    d = d[d[zc].fillna(0) <= z_max]
    traces = []
    clutter = d[d.drone_view.isna()]
    if len(clutter):
        traces.append(go.Scatter3d(x=clutter.ve, y=clutter.vn, z=clutter[zc].fillna(0), mode="markers",
                                   marker=dict(size=2, color="#56606b", opacity=0.35),
                                   name="Ruido (sin dron)", hoverinfo="skip"))
    for key, g in d[d.drone_view.notna()].groupby("drone_view"):
        col = drone_colors.get(key, "#cccccc")
        traces.append(go.Scatter3d(
            x=g.ve, y=g.vn, z=g[zc].fillna(0), mode="markers",
            marker=dict(size=3, color=col, opacity=0.85, symbol="circle"),
            name=f"Etiqueta: {key}", customdata=np.stack([g.sensor_id, g.label_status, g.label_source], -1),
            hovertemplate=(f"<b>Etiquetada: {key}</b><br>Sensor %{{customdata[0]}}"
                           "<br>%{customdata[1]} · por %{customdata[2]}<extra></extra>")))
    return traces


def comparison_figure(go, make_subplots, layer: TrackLayer, run_name: str, ds, t_now: float, drone_colors: dict,
                      bg: str, text: str, muted: str):
    """Panel inferior: error vs tiempo, cobertura por dron (qué track lo sigue) y tracks falsos."""
    run = layer.runs.get(run_name)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.38, 0.47, 0.15],
                        subplot_titles=("Error real↔estimado por dron (m)",
                                        "Cobertura: barra de color = seguido por ese track · franja roja = sin track",
                                        "Tracks falsos por instante"))
    if run is None or not run.compare:
        fig.add_annotation(text="Sin GT o sin tracks para esta iteración", showarrow=False, x=0.5, y=0.5,
                           xref="paper", yref="paper", font=dict(color=muted))
    else:
        c = float(run.compare["c"].c.iloc[0])
        pairs = run.compare["pairs"].copy()
        missed = run.compare["missed"].copy()
        inst = run.compare["instants"].copy()
        to_t = lambda s: pd.to_datetime(s, unit="ms")  # UTC sin zona, como el visor
        keys = sorted(set(pairs.drone_key) | set(missed.drone_key), key=lambda k: layer.gt_view.get(k, k))
        views = [layer.gt_view.get(k, k) for k in keys]

        def spans(ts: pd.Series):
            """Agrupa segundos consecutivos en tramos (inicio, fin)."""
            ts = ts.sort_values().to_numpy()
            if not len(ts):
                return []
            cut = np.where(np.diff(ts) > 1500)[0]
            starts = np.r_[ts[0], ts[cut + 1]]
            ends = np.r_[ts[cut], ts[-1]]
            return list(zip(starts, ends + 1000))

        for i, key in enumerate(keys):
            view = views[i]
            col = drone_colors.get(view, "#cccccc")
            p = pairs[pairs.drone_key == key].sort_values("ts_ms")
            if len(p):
                gap = p.ts_ms.diff().fillna(1000) > 1500
                y = p.dist_m.mask(gap)  # cortar la línea donde se pierde el dron
                fig.add_trace(go.Scatter(x=to_t(p.ts_ms), y=y, mode="lines", line=dict(color=col, width=1.5),
                                         name=view, legendgroup=view, connectgaps=False,
                                         hovertemplate=f"{view}<br>%{{x|%H:%M:%S}} · %{{y:.1f}} m<extra></extra>"),
                              row=1, col=1)
            # Primero los huecos (franja roja fina y translúcida), encima las barras de los tracks.
            xs, ys = [], []
            for a, b in spans(missed[missed.drone_key == key].ts_ms):
                xs += [to_t(a), to_t(b), None]
                ys += [i, i, None]
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(255,92,92,0.45)", width=4),
                                         showlegend=False, hovertemplate=f"{view}: sin track<extra></extra>"),
                              row=2, col=1)
            if len(p):
                seg = (p.track_id != p.track_id.shift()) | (p.ts_ms.diff() > 1500)
                p = p.assign(seg=seg.cumsum())
                for _, s in p.groupby("seg"):
                    tid = s.track_id.iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[to_t(s.ts_ms.iloc[0]), to_t(s.ts_ms.iloc[-1] + 1000)], y=[i, i], mode="lines",
                        line=dict(color=run.colors.get(tid, col), width=11), showlegend=False,
                        hovertemplate=f"{view}<br>Track …{tid[-6:]}<br>%{{x|%H:%M:%S}}<extra></extra>"), row=2, col=1)
        # Todas las filas con su nombre (con ejes categóricos Plotly ocultaba algunas).
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(views))), ticktext=views,
                         range=[-0.6, len(views) - 0.4], showgrid=False, row=2, col=1)
        fig.add_hline(y=c, line=dict(color=muted, dash="dot", width=1), row=1, col=1,
                      annotation_text=f"corte GOSPA {c:.0f} m", annotation_font_color=muted)
        fig.add_trace(go.Scatter(x=to_t(inst.ts_ms), y=inst.n_false, mode="lines", line=dict(color="#ffb347", width=1),
                                 fill="tozeroy", showlegend=False,
                                 hovertemplate="%{x|%H:%M:%S} · %{y} tracks falsos<extra></extra>"), row=3, col=1)
    if t_now is not None:
        fig.add_vline(x=ds.t0 + pd.Timedelta(seconds=t_now), line=dict(color=text, width=1))
    axis = dict(gridcolor="#2a3540", zerolinecolor="#3a4855", color=muted)
    fig.update_layout(paper_bgcolor=bg, plot_bgcolor=bg, font=dict(color=text, size=11),
                      margin=dict(l=120, r=150, t=28, b=24), uirevision="cmp",
                      # Leyenda a la derecha: arriba se pisaba con el título del primer gráfico.
                      legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top", font=dict(size=10),
                                  title=dict(text="Error por dron", font=dict(size=10, color=muted))),
                      hovermode="closest")
    fig.update_xaxes(axis)
    fig.update_yaxes(axis)
    fig.update_annotations(font=dict(size=11, color=muted))
    return fig


def summary_tables(layer: TrackLayer):
    """Tablas (DataFrames ya formateados) por dron e iteración, y diferencia A/B si hay dos runs."""
    out = {}
    for name, run in layer.runs.items():
        if not run.compare:
            continue
        d = run.compare["drones"].copy()
        d["Dron"] = d.drone_key.map(lambda k: layer.gt_view.get(k, k))
        out[name] = {
            "drones": d[["Dron", "pct_tracked", "err_mean_m", "err_p90_m", "n_tracks", "id_switches",
                         "first_track_delay_s"]].rename(columns={
                "pct_tracked": "% seguido", "err_mean_m": "Error medio (m)", "err_p90_m": "Error p90 (m)",
                "n_tracks": "Tracks usados", "id_switches": "Cambios de track",
                "first_track_delay_s": "Retraso 1er track (s)"}),
            "iteration": run.compare["iterations"],
        }
    return out
